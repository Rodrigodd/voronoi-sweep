#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use macroquad::prelude::*;

/// A point in 2D space. it is ordered in lexicographic order.
#[derive(PartialEq, Clone, Copy, Debug)]
pub struct Point {
    pos: Vec2,
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

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Event {
    /// A site.
    Site(Point),
    /// (p, (q, r, s)), where p is a intersection between Cqr and Crs.
    Intersection(Point, (Point, Point, Point)),
}

/// A benchline is a interleaved sequence of regions and boundaries.
pub struct Benchline {
    /// The regions and boundaries in the benchline.
    ///
    /// The regions are represented by its center point, and the boundaries are represented by
    /// bisector segments, in the form [region1, boundary1, region2, boundary2, ..., regionN].
    ///
    /// Note that the last element of the benchline is always a region, but I can't represent that
    /// cleanly in Rust type system. This will be handled by ignoring the last boundary in the
    /// Vec.
    regions: Vec<(Point, Bisector)>,
}
impl Benchline {
    fn new(first_region: Point) -> Self {
        Self {
            regions: vec![(first_region, Bisector::nill())],
        }
    }

    pub fn get_regions(&self) -> impl Iterator<Item = Point> + '_ {
        self.regions.iter().map(|(p, _)| *p)
    }

    /// Find the region that contains the point.
    ///
    /// The regions fills the entire space, so there is always a region that contains the point.
    fn find_region(&self, p: Point) -> usize {
        // debug print
        println!("find_region for {:?}", p);
        for (r, b) in self.regions.iter() {
            println!("{:?}, {:?},{:?}", r.pos, b.a, b.b);
        }

        for (i, (_, b)) in self.regions[..self.regions.len() - 1].iter().enumerate() {
            if b.star_cmp(p) == std::cmp::Ordering::Less {
                return i;
            }
        }

        self.regions.len() - 1
    }

    /// Find the index of region `r`, whose neighbors are `q` and `s`.
    fn find_region3(&self, q: Point, r: Point, s: Point) -> usize {
        println!("find_region3");
        for (r, b) in self.regions.iter() {
            println!("{:?}, {:?},{:?}", r.pos, b.a, b.b);
        }
        for (i, window) in self.regions.windows(3).enumerate() {
            if q == window[0].0 && r == window[1].0 && s == window[2].0 {
                return i + 1;
            }
        }

        unreachable!()
    }

    /// Insert a new region within the region at the given index.
    fn insert(&mut self, region_idx: usize, region: (Bisector, Point, Bisector)) {
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
    fn get_region(&self, q_idx: usize) -> Point {
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
    on_progress: &mut impl FnMut(&Benchline),
) -> HashMap<Point, Vec<Point>> {
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

    let mut vertices = sites
        .iter()
        .map(|site| (*site, Vec::new()))
        .collect::<HashMap<_, _>>();

    // 1. initialize Q with all sites
    let mut events = BinaryHeap::from_iter(sites.iter().map(|site| Reverse(Event::Site(*site))));

    // 2. p <- extract_min(Q)
    let Some(Reverse(Event::Site(p))) = events.pop() else {
        return vertices;
    };

    // 3. L <- the list containing Rp.
    let mut benchline = Benchline::new(p);

    // 4. while Q is not empty begin
    // 5. p <- extract min(Q)
    while let Some(Reverse(event)) = events.pop() {
        on_progress(&benchline);
        println!("event {:?} of {:?}", event, events);
        // 6. case
        match event {
            // 7. p is a site:
            Event::Site(p) => {
                // 8. Find an occurrence of a region Rq* on L containing p.
                let q_idx = benchline.find_region(p);
                let q = benchline.get_region(q_idx);

                // 9. Create bisector Bpq*.
                let bpq = Bisector::new(p, q);

                // 10. Update list L so that it contains ..., Rq*, Cpq-, Rp*, Cpq+, Rq*, ... in
                //     place of Rq*.
                benchline.insert(q_idx, (bpq.c_minus(), p, bpq.c_plus()));

                // 11. Delete from Q the intersection between the left and right boundary of Rq*,
                //     if any.

                // 12. Insert into Q the intersection between Cpq- and its neighbor to the left on
                //     L, if any, and the intersection between Cpq+, and its neighbor to the right,
                //     if any.
                'left: {
                    let Some(left_neighbor) = benchline.get_left_boundary(q_idx) else {
                        break 'left;
                    };
                    let Some(p) = bpq.c_minus().star_intersection(left_neighbor) else {
                        break 'left;
                    };

                    let q = benchline.get_region(q_idx - 1);
                    let r = benchline.get_region(q_idx);
                    let s = benchline.get_region(q_idx + 1);
                    events.push(Reverse(Event::Intersection(p, (q, r, s))));
                }

                'right: {
                    let Some(right_neighbor) = benchline.get_righ_boundary(q_idx + 2) else {
                        break 'right;
                    };
                    let Some(p) = bpq.c_plus().star_intersection(right_neighbor) else {
                        break 'right;
                    };

                    let q = benchline.get_region(q_idx + 1);
                    let r = benchline.get_region(q_idx + 2);
                    let s = benchline.get_region(q_idx + 3);
                    events.push(Reverse(Event::Intersection(p, (q, r, s))));
                }
            }
            // 13. p is an intersection:
            Event::Intersection(p, (q, r, s)) => {
                // 14. Let p be the intersection of boundaries Cqr and Crs.

                // 15. Create the bisector Bqs*.
                let bqs = Bisector::new(q, s);

                // 16. Update list L so it contains Cqs = Cqs- or Cqs+,as appropriate, instead of Cqr, R*r, Crs.
                let r_idx = benchline.find_region3(q, r, s);
                let cqr = if q > r { bqs.c_plus() } else { bqs.c_minus() };
                benchline.remove(r_idx, cqr);

                // 17. Delete from Q any intersection between Cqr and its neighbor to the left and between Crs and its neighbor to the right.

                // 18. Insert any intersections between Cqs and its neighbors to the left or right into Q.
                'left: {
                    let Some(left_neighbor) = benchline.get_left_boundary(r_idx - 1) else {
                        break 'left;
                    };
                    let Some(p) = cqr.star_intersection(left_neighbor) else {
                        break 'left;
                    };

                    let q = benchline.get_region(r_idx - 2);
                    let r = benchline.get_region(r_idx - 1);
                    let s = benchline.get_region(r_idx);
                    println!("left: {:?}, {:?}, {:?}", q, r, s);
                    events.push(Reverse(Event::Intersection(p, (q, r, s))));
                }

                'right: {
                    let Some(right_neighbor) = benchline.get_righ_boundary(r_idx) else {
                        break 'right;
                    };
                    let Some(p) = cqr.star_intersection(right_neighbor) else {
                        break 'right;
                    };

                    let q = benchline.get_region(r_idx - 1);
                    let r = benchline.get_region(r_idx);
                    let s = benchline.get_region(r_idx + 1);
                    println!("right: {:?}, {:?}, {:?}", q, r, s);
                    events.push(Reverse(Event::Intersection(p, (q, r, s))));
                }

                // 19. Mark p as a vertex and as an endpoint of Bqr*, Brs*, and Bqs*.
                vertices.entry(q).or_default().push(p);
                vertices.entry(r).or_default().push(p);
                vertices.entry(s).or_default().push(p);
            }
        }
    }

    on_progress(&benchline);

    // 20. end
    vertices
}

/// A segment of the bisector of two sites.
#[derive(Clone, Copy, Debug)]
struct Bisector {
    /// The higher point, the minimun point of the hyperbola "bisector*".
    a: Point,
    /// The lower point.
    b: Point,
    /// The x value of the leftmost point of the hyperbola segment.
    min_x: f32,
    /// The x value of the rightmost point of the hyperbola segment.
    max_x: f32,
}
impl Bisector {
    fn nill() -> Self {
        Self {
            a: Point { pos: Vec2::ZERO },
            b: Point { pos: Vec2::ZERO },
            min_x: 0.,
            max_x: 0.,
        }
    }

    fn new(mut a: Point, mut b: Point) -> Self {
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

    fn c_minus(self) -> Bisector {
        Bisector {
            max_x: self.a.pos.x,
            ..self
        }
    }
    fn c_plus(self) -> Bisector {
        Bisector {
            min_x: self.a.pos.x,
            ..self
        }
    }

    /// Return if point is on the left side or right side of the hyperbola, obtained by the
    /// *-mapping of the bisector.
    fn star_cmp(&self, point: Point) -> std::cmp::Ordering {
        let dx = self.b.pos.x - self.a.pos.x;
        let dy = self.b.pos.y - self.a.pos.y;

        if dy == 0.0 {
            // the bisector is a vertical line segment
            return point.pos.x.partial_cmp(&self.a.pos.x).unwrap();
        }

        // The y value of the hyperbola (bisector*) segment at x = point.x
        let bisector_star_dy = {
            let x = point.pos.x;

            let x2 = x * x;
            let dx2 = dx * dx;
            let dy2 = dy * dy;

            let sqrt = |x: f32| x.sqrt();

            sqrt(x2 + (dx2 - 2.0 * dx * x + dy2).powi(2) / (4.0 * dy2))
                + (dx2 - 2.0 * dx * x + dy2) / (2.0 * dy)
        };

        let bisector_star_y = self.a.pos.y + bisector_star_dy;

        let ord = point.pos.y.partial_cmp(&bisector_star_y).unwrap();

        // if the point is on the right side of the vertex, above y means left side.
        if point.pos.x > self.a.pos.x {
            ord.reverse()
        } else {
            ord
        }
    }

    /// Returns the intersection point of two bisectors.
    fn intersection(&self, other: Bisector) -> Option<Point> {
        if self.min_x > other.max_x || self.max_x < other.min_x {
            return None;
        }

        let px = self.a.pos.x;
        let py = self.a.pos.y;
        let qx = self.b.pos.x;
        let qy = self.b.pos.y;

        let rx = other.a.pos.x;
        let ry = other.a.pos.y;
        let sx = other.b.pos.x;
        let sy = other.b.pos.y;

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
    fn star_intersection(&self, other: Bisector) -> Option<Point> {
        self.intersection(other).map(|p| start_map(p, self.a))
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

#[macroquad::main("Tree")]
async fn main() {
    let camera = Camera2D {
        zoom: vec2(1., 1.),
        target: vec2(0.0, 0.5),
        ..Default::default()
    };

    set_camera(&camera);
    loop {
        clear_background(LIGHTGRAY);

        draw_circle(0., 0., 0.03, DARKGRAY);

        next_frame().await
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::{Arbitrary, Gen};

    impl Arbitrary for Point {
        fn arbitrary(g: &mut Gen) -> Self {
            let mut gen = || loop {
                let x = f32::arbitrary(g);
                if x.is_finite() && x.abs() < 1.0e12 {
                    return x;
                }
            };

            Point {
                pos: vec2(gen(), gen()),
            }
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < (a + b).abs() * 1e-5
    }

    #[test]
    fn bisector_intersection() {
        let a = Point { pos: vec2(0., 0.) };
        let b = Point { pos: vec2(1., 0.) };
        let c = Point { pos: vec2(0., 1.) };
        let d = Point { pos: vec2(1., 1.) };

        let ad = super::Bisector::new(a, d);
        let bc = super::Bisector::new(b, c);

        let p = ad.intersection(bc).unwrap();

        assert_eq!(p.pos, vec2(0.5, 0.5));
    }

    #[quickcheck]
    fn bisector_intersections(a: Point, b: Point, c: Point, d: Point) -> bool {
        let ab = super::Bisector::new(a, b);
        let cd = super::Bisector::new(c, d);

        println!("ab: {:?}, cd: {:?}", ab, cd);

        let Some(p) = ab.intersection(cd) else {
            return true;
        };

        println!("p: {:?}", p);

        let da = dist(a, p);
        let db = dist(b, p);
        let dc = dist(c, p);
        let dd = dist(d, p);

        // Ignore NaN/Infinite results
        if !da.is_finite() || !db.is_finite() || !dc.is_finite() || !dd.is_finite() {
            return true;
        }

        println!("da: {}, db: {}, dc: {}, dd: {}", da, db, dc, dd);

        close(da, db) && close(dc, dd)
    }

    #[test]
    fn bisector_cmp() {
        let p = Point { pos: vec2(1., 0.) };
        let q = Point { pos: vec2(0., 1.) };
        let r = Point { pos: vec2(1., 1.) };
        let t = Point { pos: vec2(-1., 1.) };

        let bpq = Bisector::new(p, q);

        assert_eq!(bpq.star_cmp(r), std::cmp::Ordering::Greater);
        assert_eq!(bpq.star_cmp(t), std::cmp::Ordering::Less);
    }

    #[test]
    fn bisector_no_intersection() {
        let p = Point { pos: vec2(1., 0.) };
        let q = Point { pos: vec2(0., 1.) };
        let r = Point { pos: vec2(1., 1.) };

        let bpq = Bisector::new(p, q);
        let bqr = Bisector::new(q, r);
        let bpr = Bisector::new(p, r);

        let cqr_plus = bqr.c_plus();
        let cqr_minus = bqr.c_minus();

        let cpq_minus = bpq.c_minus();
        assert_eq!(cpq_minus.intersection(cqr_plus), None);
        assert_eq!(cpq_minus.intersection(cqr_minus), None);

        let cpr_plus = bpr.c_plus();
        assert_eq!(cpq_minus.intersection(cpr_plus), None);
        assert_eq!(cqr_plus.intersection(cpr_plus), None);
    }

    #[test]
    fn diagram() {
        let p = Point { pos: vec2(1., 0.) };
        let q = Point { pos: vec2(0., 1.) };
        let r = Point { pos: vec2(1., 1.) };
        let points = [p, q, r];

        let mut expected_benchline: &[&[Point]] = &[
            &[p],             // first region
            &[p, q, p],       // insert q
            &[p, q, p, r, p], // insert r
            &[p, q, r, p],    // intersect q and r
        ];

        let vertexes = fortune_algorithm(&points, &mut |benchline| {
            let regions = benchline.get_regions().collect::<Vec<_>>();
            println!("regions: {:?}", regions);
            assert_eq!(regions, expected_benchline[0]);
            expected_benchline = &expected_benchline[1..];
        });

        for (site, vertexes) in vertexes {
            println!("site: {:?}", site);
            for vertex in &vertexes {
                println!("  vertex: {:?}", vertex);
            }

            let dists: Vec<f32> = vertexes.iter().map(|v| dist(site, *v)).collect();

            assert!(dists.iter().all(|d| *d > 0.0));

            // all dists should be the same
            assert!(dists.windows(2).all(|w| close(w[0], w[1])));
        }
    }

    #[quickcheck]
    fn diagram_fuzz(mut points: Vec<(u8, u8)>) {
        points.sort();
        points.dedup();

        let points = points
            .into_iter()
            .map(|(x, y)| Point {
                pos: vec2(x as f32, y as f32),
            })
            .collect::<Vec<_>>();

        fortune_algorithm(&points, &mut |_| {});
    }

    #[test]
    fn diagram1() {
        let points = [
            (4, 0), //
            (0, 1),
            (5, 2),
            (5, 3),
        ];

        let points = points
            .iter()
            .map(|(x, y)| Point {
                pos: vec2(*x as f32, *y as f32),
            })
            .collect::<Vec<_>>();

        fortune_algorithm(&points, &mut |benchline| {
            let regions = benchline.get_regions().collect::<Vec<_>>();
            println!("regions: {:?}", regions);
        });
    }
}
