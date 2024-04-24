#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

use macroquad::prelude::*;

mod heap;
use heap::Heap;

#[cfg(test)]
mod test;

/// A point in 2D space. it is ordered in lexicographic order.
#[derive(PartialEq, Clone, Copy)]
pub struct Point {
    pos: Vec2,
}
impl std::fmt::Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // f.debug_tuple("Point")
        //     .field(&self.pos.x)
        //     .field(&self.pos.y)
        //     .finish()
        match ((self.pos.x * 100.0) as u32, (self.pos.y * 100.0) as u32) {
            (400, 000) => f.write_str("p"),
            (000, 100) => f.write_str("q"),
            (500, 200) => f.write_str("r"),
            (500, 300) => f.write_str("s"),
            _ => f
                .debug_tuple("Point")
                .field(&self.pos.x)
                .field(&self.pos.y)
                .finish(),
        }
    }
}
impl std::hash::Hash for Point {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pos.x.to_bits().hash(state);
        self.pos.y.to_bits().hash(state);
    }
}
impl Eq for Point {}
impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Point {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.pos.y == other.pos.y {
            self.pos.x.partial_cmp(&other.pos.x).unwrap()
        } else {
            self.pos.y.partial_cmp(&other.pos.y).unwrap()
        }
    }
}

type SiteIdx = u32;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
pub enum Event {
    /// A site.
    Site(SiteIdx),
    /// (p, (q, r, s)), where p is the intersection point of Cqr* and Crs*.
    Intersection(Point, (SiteIdx, SiteIdx, SiteIdx)),
}

/// A benchline is a interleaved sequence of regions and boundaries.
#[derive(Clone)]
pub struct Benchline {
    /// The regions and boundaries in the benchline.
    ///
    /// The regions are represented by its center point, and the boundaries are represented by
    /// bisector segments, in the form [region1, boundary1, region2, boundary2, ..., regionN].
    ///
    /// Note that the last element of the benchline is always a region, but I can't represent that
    /// cleanly in Rust type system. This will be handled by ignoring the last boundary in the
    /// Vec.
    regions: Vec<(SiteIdx, Bisector)>,
}
impl Benchline {
    fn new(first_region: SiteIdx) -> Self {
        Self {
            regions: vec![(first_region, Bisector::nill())],
        }
    }

    pub fn get_regions(&self) -> impl Iterator<Item = SiteIdx> + '_ {
        self.regions.iter().map(|(p, _)| *p)
    }

    /// Find the region that contains the point.
    ///
    /// The regions fills the entire space, so there is always a region that contains the point.
    fn find_region(&self, sites: &[Point], p: Point) -> usize {
        // debug print
        // println!("find_region for {:?}", p);
        // for (r, b) in self.regions.iter() {
        //     println!("{:?}, {:?},{:?}", r.pos, b.a, b.b);
        // }

        for (i, (_, b)) in self.regions[..self.regions.len() - 1].iter().enumerate() {
            if b.star_cmp(sites, p) == std::cmp::Ordering::Less {
                return i;
            }
        }

        self.regions.len() - 1
    }

    /// Find the index of region `r`, whose neighbors are `q` and `s`.
    fn find_region3(&self, q: SiteIdx, r: SiteIdx, s: SiteIdx) -> usize {
        // println!("find_region3 {:?} {:?} {:?}", q, r, s);
        // for (r, b) in self.regions.iter() {
        //     println!("{:?}, {:?},{:?}", r.pos, b.a, b.b);
        // }

        for (i, window) in self.regions.windows(3).enumerate() {
            if q == window[0].0 && r == window[1].0 && s == window[2].0 {
                return i + 1;
            }
        }

        unreachable!()
    }

    /// Insert a new region within the region at the given index.
    fn insert(&mut self, region_idx: usize, region: (Bisector, SiteIdx, Bisector)) {
        let (p, h) = self.regions[region_idx];

        let (hl, q, hr) = region;
        self.regions
            .splice(region_idx..=region_idx, [(p, hl), (q, hr), (p, h)]);
    }

    /// Remove the region at the given index and replace it with the boundary of it neighbor
    /// regions.
    fn remove(&mut self, region_idx: usize, boundary: Bisector) {
        self.regions.remove(region_idx);
        self.regions[region_idx - 1].1 = boundary
    }

    /// Return the region at the given index.
    fn get_region(&self, q_idx: usize) -> SiteIdx {
        self.regions[q_idx].0
    }

    /// Return the right boundary of the region at the given index .
    fn get_righ_boundary(&self, q_idx: usize) -> Option<Bisector> {
        // the last region has no right boundary
        if q_idx == self.regions.len() - 1 {
            return None;
        }

        Some(self.regions[q_idx].1)
    }

    fn get_left_boundary(&self, q_idx: usize) -> Option<Bisector> {
        // the first region has no left boundary
        if q_idx == 0 {
            return None;
        }

        Some(self.regions[q_idx - 1].1)
    }
}

/// Fortune algorithm.
///
/// Reference:
/// - S.J. Fortune, A sweepline algorithm for Voronoi diagrams, Algorithmica 2 (1987) 153–174.
pub fn fortune_algorithm(
    sites: &[Point],
    on_progress: &mut impl FnMut(&Benchline, &[Event]),
) -> Vec<Vec<Point>> {
    // Algorithm 1: Computation of V*(S).
    // Input:
    //  - S is a set of n >= 1 points with unique bottommost point.
    // Output:
    //  - The bisectors and vertices of V*.
    // Data structures:
    //  - Q: a priority queue of points in the plane, ordered lexicographically. Each point is
    //  labeled as a site, or labeled as the intersection of a pair of boundaries of a single
    //  region. Q may contain duplicate instances of the same point with distinct labels; the
    //  ordering of duplicates is irrelevant.
    //  - L: a sequence (r1, c1, r2, . . . , rk) of regions (labeled by site) and boundaries
    //  (labeled by a pair of sites). Note that a region can appear many times on L.

    let mut vertices = vec![Vec::new(); sites.len()];

    // 1. initialize Q with all sites
    let mut events = Heap::new(|a: &Event, b: &Event| {
        let pa = match a {
            Event::Site(p) => sites[*p as usize],
            Event::Intersection(p, _) => *p,
        };
        let pb = match b {
            Event::Site(p) => sites[*p as usize],
            Event::Intersection(p, _) => *p,
        };
        pa.cmp(&pb)
    });

    // 2. p <- extract_min(Q)
    let Some(Event::Site(p)) = events.pop() else {
        return vertices;
    };

    // 3. L <- the list containing Rp.
    let mut benchline = Benchline::new(p);

    on_progress(&benchline, events.as_slice());

    // 4. while Q is not empty begin
    // 5. p <- extract min(Q)
    while let Some(event) = events.pop() {
        println!("event {:?} of {:?}", event, events);
        // 6. case
        match event {
            // 7. p is a site:
            Event::Site(p_idx) => {
                let p = sites[p_idx as usize];
                // 8. Find an occurrence of a region Rq* on L containing p.
                let reg_q_idx = benchline.find_region(sites, p);
                let q_idx = benchline.get_region(reg_q_idx);

                // 9. Create bisector Bpq*.
                let bpq = Bisector::new(p_idx, q_idx);

                // 10. Update list L so that it contains ..., Rq*, Cpq-, Rp*, Cpq+, Rq*, ... in
                //     place of Rq*.
                benchline.insert(reg_q_idx, (bpq.c_minus(sites), p_idx, bpq.c_plus(sites)));

                // 11. Delete from Q the intersection between the left and right boundary of Rq*,
                //     if any.

                // 12. Insert into Q the intersection between Cpq- and its neighbor to the left on
                //     L, if any, and the intersection between Cpq+, and its neighbor to the right,
                //     if any.
                'left: {
                    let Some(left_neighbor) = benchline.get_left_boundary(reg_q_idx) else {
                        break 'left;
                    };
                    let Some(p) = bpq.c_minus(sites).star_intersection(sites, left_neighbor) else {
                        break 'left;
                    };

                    let q = benchline.get_region(reg_q_idx - 1);
                    let r = benchline.get_region(reg_q_idx);
                    let s = benchline.get_region(reg_q_idx + 1);
                    events.push(Event::Intersection(p, (q, r, s)));
                    println!("l intersection {:?}, ({:?}, {:?}, {:?})", p, q, r, s);
                }

                'right: {
                    let Some(right_neighbor) = benchline.get_righ_boundary(reg_q_idx + 2) else {
                        break 'right;
                    };
                    let Some(p) = bpq.c_plus(sites).star_intersection(sites, right_neighbor) else {
                        break 'right;
                    };

                    let q = benchline.get_region(reg_q_idx + 1);
                    let r = benchline.get_region(reg_q_idx + 2);
                    let s = benchline.get_region(reg_q_idx + 3);
                    events.push(Event::Intersection(p, (q, r, s)));
                    println!("r intersection {:?}, ({:?}, {:?}, {:?})", p, q, r, s);
                }
            }
            // 13. p is an intersection:
            Event::Intersection(p, (q_idx, r_idx, s_idx)) => {
                let q = sites[q_idx as usize];
                let r = sites[r_idx as usize];
                let s = sites[s_idx as usize];

                // 14. Let p be the intersection of boundaries Cqr and Crs.

                // 15. Create the bisector Bqs*.
                let bqs = Bisector::new(q_idx, s_idx);

                // 16. Update list L so it contains Cqs = Cqs- or Cqs+,as appropriate, instead of Cqr, R*r, Crs.
                let region_r_idx = benchline.find_region3(q_idx, r_idx, s_idx);
                let cqs = if p.pos.x < r.pos.x {
                    bqs.c_minus(sites)
                } else {
                    bqs.c_plus(sites)
                };

                let cqr = benchline.get_left_boundary(region_r_idx).unwrap();
                let crs = benchline.get_righ_boundary(region_r_idx).unwrap();

                benchline.remove(region_r_idx, cqs);

                // 17. Delete from Q any intersection between Cqr and its neighbor to the left and between Crs and its neighbor to the right.
                'left: {
                    let Some(left_neighbor) = benchline.get_left_boundary(region_r_idx - 1) else {
                        break 'left;
                    };

                    events.remove(|e| match e {
                        &Event::Intersection(_, (a, b, c)) => {
                            let retain = !((a == left_neighbor.a || a == left_neighbor.b)
                                && (b == cqr.a || b == cqr.b)
                                && (c == cqr.a || c == cqr.b));

                            if !retain {
                                println!("removing {:?}", e);
                            }

                            retain
                        }
                        _ => true,
                    });
                }

                'right: {
                    let Some(right_neighbor) = benchline.get_righ_boundary(region_r_idx) else {
                        break 'right;
                    };

                    events.remove(|e| match e {
                        &Event::Intersection(_, (a, b, c)) => {
                            let retain = !((a == crs.a || a == crs.b)
                                && (b == crs.a || b == crs.b)
                                && (c == right_neighbor.a || c == right_neighbor.b));

                            if !retain {
                                println!("removing {:?}", e);
                            }

                            retain
                        }
                        _ => true,
                    });
                }

                // 18. Insert any intersections between Cqs and its neighbors to the left or right into Q.
                'left: {
                    let Some(left_neighbor) = benchline.get_left_boundary(region_r_idx - 1) else {
                        break 'left;
                    };
                    let Some(p) = cqs.star_intersection(sites, left_neighbor) else {
                        break 'left;
                    };

                    let q = benchline.get_region(region_r_idx - 2);
                    let r = benchline.get_region(region_r_idx - 1);
                    let s = benchline.get_region(region_r_idx);
                    events.push(Event::Intersection(p, (q, r, s)));
                    println!("il intersection {:?}, ({:?}, {:?}, {:?})", p, q, r, s);
                }

                'right: {
                    let Some(right_neighbor) = benchline.get_righ_boundary(region_r_idx) else {
                        break 'right;
                    };
                    let Some(p) = cqs.star_intersection(sites, right_neighbor) else {
                        break 'right;
                    };

                    let q = benchline.get_region(region_r_idx - 1);
                    let r = benchline.get_region(region_r_idx);
                    let s = benchline.get_region(region_r_idx + 1);
                    events.push(Event::Intersection(p, (q, r, s)));
                    println!("ir intersection {:?}, ({:?}, {:?}, {:?})", p, q, r, s);
                }

                // 19. Mark p as a vertex and as an endpoint of Bqr*, Brs*, and Bqs*.
                let p_unstar = circumcenter(q, r, s);
                vertices[q_idx as usize].push(p_unstar);
                vertices[r_idx as usize].push(p_unstar);
                vertices[s_idx as usize].push(p_unstar);
            }
        }
        on_progress(&benchline, events.as_slice());
    }

    // 20. end
    vertices
}

/// A segment of the bisector of two sites.
#[derive(Clone, Copy, Debug)]
struct Bisector {
    /// The higher point, the minimun point of the hyperbola "bisector*".
    a: SiteIdx,
    /// The lower point.
    b: SiteIdx,
    /// The x value of the leftmost point of the hyperbola segment.
    min_x: f32,
    /// The x value of the rightmost point of the hyperbola segment.
    max_x: f32,
}
impl Bisector {
    fn nill() -> Self {
        Self {
            a: SiteIdx::MAX,
            b: SiteIdx::MAX,
            min_x: 0.,
            max_x: 0.,
        }
    }

    fn new(mut a: SiteIdx, mut b: SiteIdx) -> Self {
        if a < b {
            std::mem::swap(&mut a, &mut b);
        }
        Self {
            a,
            b,
            min_x: f32::NEG_INFINITY,
            max_x: f32::INFINITY,
        }
    }

    fn c_minus(self, sites: &[Point]) -> Bisector {
        let a = sites[self.a as usize];
        Bisector {
            max_x: a.pos.x,
            ..self
        }
    }
    fn c_plus(self, sites: &[Point]) -> Bisector {
        let a = sites[self.a as usize];
        Bisector {
            min_x: a.pos.x,
            ..self
        }
    }

    /// The y value of the line bisector at x.
    fn y_at(&self, sites: &[Point], x: f32) -> f32 {
        let (a, b) = self.ab(sites);

        let x = x - a.pos.x;
        let dx = b.pos.x - a.pos.x;
        let dy = b.pos.y - a.pos.y;

        let dx2 = dx * dx;
        let dy2 = dy * dy;

        a.pos.y + (dx2 - 2.0 * dx * x + dy2) / (2.0 * dy)
        // 1 + (1 + 2*0.5 + 1) / (-2*1)
        // 1 + (1 + 1 + 1) / -2 = 1 - 1.5 = -0.5
    }

    /// The y value of the hyperbola "bisector*" at x.
    fn y_star_at(&self, sites: &[Point], x: f32) -> f32 {
        let (a, b) = self.ab(sites);

        let dx = b.pos.x - a.pos.x;
        let dy = b.pos.y - a.pos.y;
        let x = x - a.pos.x;

        let x2 = x * x;
        let dx2 = dx * dx;
        let dy2 = dy * dy;

        let sqrt = |x: f32| x.sqrt();

        a.pos.y
            + sqrt(x2 + (dx2 - 2.0 * dx * x + dy2).powi(2) / (4.0 * dy2))
            + (dx2 - 2.0 * dx * x + dy2) / (2.0 * dy)
    }

    /// Return if point is on the left side or right side of the hyperbola, obtained by the
    /// *-mapping of the bisector.
    fn star_cmp(&self, sites: &[Point], point: Point) -> std::cmp::Ordering {
        let (a, b) = self.ab(sites);

        if point.pos.x < self.min_x {
            return std::cmp::Ordering::Less;
        }
        if point.pos.x >= self.max_x {
            return std::cmp::Ordering::Greater;
        }

        if (b.pos.y - a.pos.y) == 0.0 {
            // the bisector is a vertical line segment
            return point.pos.x.partial_cmp(&a.pos.x).unwrap();
        }

        let bisector_star_y = self.y_star_at(sites, point.pos.x);

        let ord = point.pos.y.partial_cmp(&bisector_star_y).unwrap();

        // if this is the right half of the hyperbola, above y means left side.
        if self.min_x == a.pos.x {
            ord.reverse()
        } else {
            ord
        }
    }

    /// Returns the intersection point of two bisectors.
    fn intersection(&self, sites: &[Point], other: Bisector) -> Option<Point> {
        let (a, b) = self.ab(sites);
        let (oa, ob) = other.ab(sites);

        if self.min_x > other.max_x || self.max_x < other.min_x {
            return None;
        }

        let px = a.pos.x;
        let py = a.pos.y;
        let qx = b.pos.x;
        let qy = b.pos.y;

        let rx = oa.pos.x;
        let ry = oa.pos.y;
        let sx = ob.pos.x;
        let sy = ob.pos.y;

        let px2 = px * px;
        let py2 = py * py;
        let qx2 = qx * qx;
        let qy2 = qy * qy;
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let sx2 = sx * sx;
        let sy2 = sy * sy;

        let d = 2.0 * px * ry - 2.0 * px * sy - 2.0 * py * rx + 2.0 * py * sx - 2.0 * qx * ry
            + 2.0 * qx * sy
            + 2.0 * qy * rx
            - 2.0 * qy * sx;

        if d == 0.0 {
            return None;
        }

        let nx =
            px2 * ry - px2 * sy + py2 * ry - py2 * sy - py * rx2 - py * ry2 + py * sx2 + py * sy2
                - qx2 * ry
                + qx2 * sy
                - qy2 * ry
                + qy2 * sy
                + qy * rx2
                + qy * ry2
                - qy * sx2
                - qy * sy2;
        let ny = -px2 * rx + px2 * sx + px * rx2 + px * ry2 - px * sx2 - px * sy2 - py2 * rx
            + py2 * sx
            + qx2 * rx
            - qx2 * sx
            - qx * rx2
            - qx * ry2
            + qx * sx2
            + qx * sy2
            + qy2 * rx
            - qy2 * sx;

        let x = nx / d;
        let y = ny / d;

        if x < self.min_x || x > self.max_x || x < other.min_x || x > other.max_x {
            return None;
        }

        let p = Point { pos: vec2(x, y) };
        Some(p)
    }

    /// Returns the intersection point of two bisectors, *-mapped.
    fn star_intersection(&self, sites: &[Point], other: Bisector) -> Option<Point> {
        let (a, _) = self.ab(sites);
        self.intersection(sites, other).map(|p| start_map(p, a))
    }

    fn ab(&self, sites: &[Point]) -> (Point, Point) {
        (sites[self.a as usize], sites[self.b as usize])
    }
}

fn dist(a: Point, b: Point) -> f32 {
    let dx = a.pos.x - b.pos.x;
    let dy = a.pos.y - b.pos.y;
    (dx * dx + dy * dy).sqrt()
}

/// Returns the *-mapping of the point `p` with respect to the region of site `q`.
///
/// *(x,y) = (x, y + dist(q, (x, y))
fn start_map(p: Point, q: Point) -> Point {
    let x = p.pos.x;
    let y = p.pos.y;
    Point {
        pos: vec2(x, y + dist(q, p)),
    }
}

/// Finds the circumcenter of the triangle formed by the points `a`, `b`, and `c`.
fn circumcenter(a: Point, b: Point, c: Point) -> Point {
    let d = 2.0
        * (a.pos.x * (b.pos.y - c.pos.y)
            + b.pos.x * (c.pos.y - a.pos.y)
            + c.pos.x * (a.pos.y - b.pos.y));

    let ux = (a.pos.x * a.pos.x + a.pos.y * a.pos.y) * (b.pos.y - c.pos.y)
        + (b.pos.x * b.pos.x + b.pos.y * b.pos.y) * (c.pos.y - a.pos.y)
        + (c.pos.x * c.pos.x + c.pos.y * c.pos.y) * (a.pos.y - b.pos.y);
    let uy = (a.pos.x * a.pos.x + a.pos.y * a.pos.y) * (c.pos.x - b.pos.x)
        + (b.pos.x * b.pos.x + b.pos.y * b.pos.y) * (a.pos.x - c.pos.x)
        + (c.pos.x * c.pos.x + c.pos.y * c.pos.y) * (b.pos.x - a.pos.x);

    Point {
        pos: vec2(ux / d, uy / d),
    }
}

#[macroquad::main("Tree")]
async fn main() {
    let (send, recv) = std::sync::mpsc::sync_channel(0);

    let points = [
        (4, 0), //
        (0, 1),
        (5, 2),
        (5, 3),
    ];

    let sites = &points
        .iter()
        .map(|(x, y)| Point {
            pos: vec2(*x as f32, *y as f32),
        })
        .collect::<Vec<_>>();

    let mut _thread = Some(std::thread::spawn({
        let sites = sites.clone();
        move || {
            fortune_algorithm(&sites, &mut |benchline, events| {
                send.send((benchline.clone(), events.to_vec())).unwrap();
            })
        }
    }));

    let width = points.iter().map(|(x, _)| *x).max().unwrap() as f32;
    let height = points.iter().map(|(_, y)| *y).max().unwrap() as f32;

    let max = width.max(height);
    let width = max;
    let height = max;

    let camera = Camera2D {
        zoom: vec2(1.0 / width, -1.0 / height),
        target: vec2(width / 2.0, height / 2.0),
        ..Default::default()
    };

    let (mut benchline, mut events) = recv.recv().unwrap();

    set_camera(&camera);

    let bottom_left = camera.screen_to_world(vec2(0.0, 0.0));
    let top_right = camera.screen_to_world(vec2(screen_width(), screen_height()));
    let left = bottom_left.x;
    let right = top_right.x;
    let top = bottom_left.y;
    let bottom = top_right.y;

    let mut vertexes = None;

    loop {
        clear_background(LIGHTGRAY);

        for point in points {
            draw_circle(point.0 as f32, point.1 as f32, 0.1, RED);
        }

        if is_key_pressed(KeyCode::Space) {
            if let Ok(b) = recv.try_recv() {
                (benchline, events) = b;
            }

            if _thread.as_ref().is_some_and(|x| x.is_finished()) {
                vertexes = Some(_thread.take().unwrap().join().unwrap());
            }
        }

        if let Some(vertexes) = &vertexes {
            for vs in vertexes {
                for v in vs {
                    draw_circle(v.pos.x, v.pos.y, 0.05, RED);
                }
            }
        }

        let mut hyperbola_colors = [GREEN, GREEN];
        for (p, b) in benchline.regions.iter() {
            let p = sites[*p as usize];

            hyperbola_colors.swap(0, 1);
            let hcolor = hyperbola_colors[0];

            draw_circle(p.pos.x, p.pos.y, 0.1, BLUE);

            let step = (right - left) / 100.0;

            let y0 = b.y_at(sites, left);
            let y1 = b.y_at(sites, right);

            draw_line(left, y0, right, y1, 0.02, BLUE);
            // draw_line(b.a.pos.x, b.a.pos.y, b.b.pos.x, b.b.pos.y, 0.02, BLUE);

            for i in 0..100 {
                let mut x1 = left + i as f32 * step;
                let mut x2 = left + (i + 1) as f32 * step;

                if x2 < b.min_x {
                    continue;
                }
                if x1 < b.min_x {
                    x1 = b.min_x;
                }
                if x1 > b.max_x {
                    continue;
                }
                if x2 > b.max_x {
                    x2 = b.max_x;
                }

                let y1 = b.y_star_at(sites, x1);
                let y2 = b.y_star_at(sites, x2);
                if !y1.is_finite() || !y2.is_finite() {
                    continue;
                }
                if (y1 > top || y1 < bottom) && (y2 > top || y2 < bottom) {
                    continue;
                }
                draw_line(x1, y1, x2, y2, 0.03, hcolor);
            }
        }

        for v in events.iter() {
            match v {
                Event::Site(p) => {
                    let p = &sites[*p as usize];
                    draw_circle(p.pos.x, p.pos.y, 0.05, BLACK);
                }
                Event::Intersection(p, _) => {
                    draw_circle(p.pos.x, p.pos.y, 0.05, BLACK);
                    draw_line(p.pos.x, bottom, p.pos.x, top, 0.02, BLACK);
                }
            }
        }

        next_frame().await
    }
}
