use std::cmp::Ordering;

pub mod btree;
mod heap;

use heap::Heap;

#[cfg(test)]
mod test;

#[macro_export]
macro_rules! debugln {
    ($($arg:tt)*) => {
        #[cfg(not(coverage))]
        if false {
            println!($($arg)*)
        }
    };
}

/// A point in 2D space. it is ordered in lexicographic order.
#[derive(PartialEq, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}
impl std::ops::Sub for Point {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
impl std::ops::Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl std::ops::Mul<f32> for Point {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}
impl Point {
    /// A Point whose x and y are both NaN.
    pub const NAN: Self = Self::new(f32::NAN, f32::NAN);

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Check if both coordinates are NaN.
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() && self.y.is_nan()
    }

    pub fn perp_dot(&self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }
}
impl std::fmt::Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Point")
            .field(&self.x)
            .field(&self.y)
            .finish()
    }
}
impl std::hash::Hash for Point {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}
impl Eq for Point {}
impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Point {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.y == other.y {
            self.x.partial_cmp(&other.x).unwrap()
        } else {
            self.y.partial_cmp(&other.y).unwrap()
        }
    }
}

pub type SiteIdx = u32;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
pub enum Event {
    /// A site.
    Site(SiteIdx),
    /// (p, (q, r, s)), where p is the intersection point of Cqr* and Crs*.
    Intersection(Point, (SiteIdx, SiteIdx, SiteIdx)),
}
impl Event {
    // Return the position where this event happens.
    pub fn pos(&self, sites: &[Point]) -> Point {
        match self {
            Event::Site(p) => sites[*p as usize],
            Event::Intersection(p, _) => *p,
        }
    }
}

/// A benchline is a interleaved sequence of regions and boundaries.
#[derive(Clone)]
pub struct Beachline {
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
impl std::fmt::Debug for Beachline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.regions[0..self.regions.len() - 1]
            .iter()
            .fold(&mut f.debug_list(), |f, (p, b)| f.entry(&p).entry(&b))
            .entry(&self.regions.last().unwrap().0)
            .finish()
    }
}
impl Beachline {
    fn new(first_region: SiteIdx) -> Self {
        Self {
            regions: vec![(first_region, Bisector::nill())],
        }
    }

    pub fn get_sites(&self) -> impl Iterator<Item = SiteIdx> + '_ {
        self.regions.iter().map(|(p, _)| *p)
    }

    pub fn get_bisectors(&self) -> impl Iterator<Item = Bisector> + '_ {
        self.regions[..self.regions.len() - 1]
            .iter()
            .map(|(_, b)| *b)
    }

    /// Find the region that contains the point.
    ///
    /// The regions fills the entire space, so there is always a region that contains the point.
    ///
    /// ## From the paper:
    ///
    /// > The search can be implemented as a binary search on list L, since L contains the regions
    /// and boundaries in order on the horizontal line. If the site actually falls on a boundary,
    /// the search can return the region on either side of the boundary. Note that the actual
    /// x-coordinate where a boundary intersects the horizontal line is determined by the
    /// y-coordinate of the line.
    pub fn find_region(&self, sites: &[Point], p: Point) -> usize {
        // debug print
        // debugln!("find_region for {:?}", p);
        // for (r, b) in self.regions.iter() {
        //     debugln!("{:?}, {:?},{:?}", r, b.a, b.b);
        // }

        for (i, (_, b)) in self.regions[..self.regions.len() - 1].iter().enumerate() {
            if b.star_cmp(sites, p) == Ordering::Less {
                return i;
            }
        }

        self.regions.len() - 1
    }

    /// Find the index of region `r`, whose neighbors are `q` and `s`.
    pub fn find_region3(&self, q: SiteIdx, r: SiteIdx, s: SiteIdx) -> Option<usize> {
        debugln!("find_region3 {:?} {:?} {:?}", q, r, s);
        // for (r, b) in self.regions.iter() {
        //     debugln!("{:?}, {:?},{:?}", r, b.a, b.b);
        // }

        for (i, window) in self.regions.windows(3).enumerate() {
            if q == window[0].0 && r == window[1].0 && s == window[2].0 {
                return Some(i + 1);
            }
        }

        None
    }

    /// Insert a new region at the exactly right of the region at the given index.
    fn split2(&mut self, region_idx: usize, region: (Bisector, SiteIdx)) {
        let (p, mut h) = self.regions[region_idx];
        let (hl, q) = region;

        if h.a == p {
            h.a = q;
        } else {
            h.b = q;
        }

        self.regions
            .splice(region_idx..=region_idx, [(p, hl), (q, h)]);
    }

    /// Insert a new region within the region at the given index.
    fn split3(&mut self, region_idx: usize, region: (Bisector, SiteIdx, Bisector)) {
        let (p, h) = self.regions[region_idx];

        let (hl, q, hr) = region;
        self.regions
            .splice(region_idx..=region_idx, [(p, hl), (q, hr), (p, h)]);
    }

    /// Remove the region at the given index and replace it with the boundary of it neighbor
    /// regions.
    fn merge(&mut self, region_idx: usize, boundary: Bisector) {
        self.regions.remove(region_idx);
        self.regions[region_idx - 1].1 = boundary
    }

    /// Return the number of regions in the benchline.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Return the region at the given index.
    pub fn get_region(&self, q_idx: usize) -> SiteIdx {
        self.regions[q_idx].0
    }

    /// Return the right boundary of the region at the given index .
    pub fn get_right_boundary(&self, q_idx: usize) -> Option<Bisector> {
        // the last region has no right boundary
        if q_idx == self.regions.len() - 1 {
            return None;
        }

        Some(self.regions[q_idx].1)
    }

    pub fn get_left_boundary(&self, q_idx: usize) -> Option<Bisector> {
        // the first region has no left boundary
        if q_idx == 0 {
            return None;
        }

        Some(self.regions[q_idx - 1].1)
    }
}

/// Fortune algorithm. Based primarily on the paper "A sweepline algorithm for Voronoi diagrams" by
/// Steven Fortune, but also referencing the paper "An Efficient Implementation of Fortune's
/// Plane-Sweep Algorithm for Voronoi Diagrams" by Kenny Wong and Hausi A. Müller.
///
/// I didn't read those papers in their entirety, so I'm not sure if they specify the following:
/// points, but I needed to introduce them to solve some edge cases:
/// - The domain of Cqr- are the points to the left of the point `max(q, r)`, *excluding* the point
/// itself. The domain of Cqr+ are the points to the right of the point `min(q, r)`, *including* the
/// point itself. Fortune's paper suggests that both boundaries should include the point, and
/// Wong's paper suggests that both exclude the point, but not very clearly, as far as I read.
/// - If a point is on the hyperbola, it should be considered on the left side of the hyperbola.
/// This is needed with the previous point in mind. Fortune's paper says that the decision doesn't
/// matter, and I didn't read enough of Wong's paper to know if it is specified.
///
/// References:
/// - S.J. Fortune, A sweepline algorithm for Voronoi diagrams, Algorithmica 2 (1987), 153–174.
/// - Kenny Wong, Hausi A. Müller, An Efficient Implementation of Fortune's Plane-Sweep Algorithm for Voronoi Diagrams
pub fn fortune_algorithm(
    sites: &[Point],
    on_progress: &mut impl FnMut(&Beachline, &[Event], &[Cell]),
) -> Vec<Cell> {
    // Algorithm 1: Computation of V*(S).
    // Input:
    //  - S is a set of n >= 1 points w̵̶̵i̵̶̵t̵̶̵h̵̶̵ ̵u̵̶̵n̵̶̵i̵̶̵q̵̶̵u̵̶̵e̵̶̵ ̵b̵̶̵o̵̶̵t̵̶̵t̵̶̵o̵̶̵m̵̶̵m̵̶̵o̵̶̵s̵̶̵t̵̶̵ ̵p̵̶̵o̵̶̵i̵̶̵n̵̶̵t̵̶̵.̵
    // Output:
    //  - The bisectors and vertices of V*.
    // Data structures:
    //  - Q: a priority queue of points in the plane, ordered lexicographically. Each point is
    //  labeled as a site, or labeled as the intersection of a pair of boundaries of a single
    //  region. Q may contain duplicate instances of the same point with distinct labels; the
    //  ordering of duplicates is irrelevant.
    //  - L: a sequence (r1, c1, r2, . . . , rk) of regions (labeled by site) and boundaries
    //  (labeled by a pair of sites). Note that a region can appear many times on L.

    let mut cells = vec![Cell::new(); sites.len()];

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
        let ord = pa.cmp(&pb);

        if ord == Ordering::Equal {
            return match (a, b) {
                (Event::Site(a), Event::Site(b)) => a.cmp(b),
                (Event::Site(_), Event::Intersection(_, _)) => Ordering::Less,
                (Event::Intersection(_, _), Event::Site(_)) => Ordering::Greater,
                (&Event::Intersection(p1, (q, r, s)), &Event::Intersection(p2, (t, u, v))) => {
                    // If two intersections are coincident, the one which will merge to a Cqs+
                    // boundary will have priority. Otherwise, the Cqs- will remove the other
                    // intersection, and will not intersect again, due to its exclusive point.
                    // This does not solve the problem in case both are negative, but I think this
                    // will never happen.
                    let c1 = Bisector::c_merge(sites, p1, q, r, s);
                    let c2 = Bisector::c_merge(sites, p2, t, u, v);

                    let is_plus1 = c1.min_x.is_finite();
                    let is_plus2 = c2.min_x.is_finite();

                    match (is_plus1, is_plus2) {
                        (true, false) => Ordering::Less,
                        (false, true) => Ordering::Greater,
                        (true, true) => (q, r, s).cmp(&(t, u, v)),
                        (false, false) => {
                            // debug_assert!(false, "this case does not happen! I think...");
                            (q, r, s).cmp(&(t, u, v))
                        }
                    }
                }
            };
        }

        ord
    });

    for site in 0..sites.len() {
        events.push(Event::Site(site as SiteIdx));
    }

    // 2. p <- extract_min(Q)
    let Some(Event::Site(p)) = events.pop() else {
        return cells;
    };

    // 3. L <- the list containing Rp.
    let mut benchline = Beachline::new(p);

    // Steps 2 and 3 make only work if there is only a single bottommost point. We can modify this
    // step a little to make it work with multiple bottommost points.

    // From Kenny Wong paper, create initial vertical boundary rays Rp1, C0_p1p2, Rp2, C0_p2p3, ...
    while let Some(&Event::Site(q)) = events.peek() {
        if sites[q as usize].y != sites[p as usize].y {
            break;
        }
        events.pop();

        let Some(p) = benchline.regions.last().map(|(p, _)| *p) else {
            unreachable!();
        };

        benchline.regions.last_mut().unwrap().1 = Bisector::new(sites, p, q).c_plus(sites);
        benchline.regions.push((q, Bisector::nill()));

        cells[p as usize].add_neighbor(sites, p, q);
        cells[q as usize].add_neighbor(sites, q, p);
    }
    debugln!("initial benchline: {:?}", benchline);

    on_progress(&benchline, events.as_slice(), &cells);

    // 4. while Q is not empty begin
    // 5. p <- extract min(Q)
    while let Some(event) = events.pop() {
        debugln!("event {:?} of {:?} in {:?}", event, events, benchline);
        // 6. case
        match event {
            // 7. p is a site:
            Event::Site(p_idx) => 'site: {
                let p = sites[p_idx as usize];
                // 8. Find an occurrence of a region Rq* on L containing p.
                let reg_q_idx = benchline.find_region(sites, p);
                let q_idx = benchline.get_region(reg_q_idx);

                cells[p_idx as usize].add_neighbor(sites, p_idx, q_idx);
                cells[q_idx as usize].add_neighbor(sites, q_idx, p_idx);

                // (added) 8.1. If p and q coincide, consider them to be side by side, adding a
                //              vertical boundary between them. ..., Rq*, Cpq0, Rp*, Cpr, ... in
                //              place of Rq*, Cqr.
                if sites[p_idx as usize] == sites[q_idx as usize] {
                    debugln!("duplicated points!!");
                    let bpq = Bisector::new(sites, p_idx, q_idx);

                    let cqr = benchline.get_right_boundary(reg_q_idx).unwrap();
                    benchline.split2(reg_q_idx, (bpq, p_idx));

                    // 8.2. Add vertex for p, q and the surrounding region
                    let r = benchline.get_region(reg_q_idx + 2);
                    let vy = Bisector::new(sites, p_idx, r).y_at(sites, p.x);
                    let v = Point::new(p.x, vy);
                    debugln!("vertex {} {} {}: {:?}", p_idx, q_idx, r, v);
                    cells[p_idx as usize].add_vertex(sites, v, p_idx, q_idx, r);
                    cells[q_idx as usize].add_vertex(sites, v, q_idx, p_idx, r);
                    cells[r as usize].add_vertex(sites, v, r, p_idx, q_idx);

                    // 8.3: Replace from Q the intersection between Cqr+ and its neighbor to the
                    //      right, with the intersection between Cpr+ and its neighbor to the right
                    //      (will be at the same point).
                    'right: {
                        let Some(right_neighbor) = benchline.get_right_boundary(reg_q_idx + 2)
                        else {
                            break 'right;
                        };

                        debugln!("cqr   {:?}", cqr);
                        debugln!("right neighbor {:?}", right_neighbor);

                        if let Some(Event::Intersection(p, (a, b, c))) =
                            events.find_mut(|e| match e {
                                &Event::Intersection(_, (a, b, c)) => {
                                    let found = (a == cqr.a || a == cqr.b)
                                        && (b == cqr.a || b == cqr.b)
                                        && (c == right_neighbor.a || c == right_neighbor.b);

                                    if found {
                                        debugln!("removing {:?}", e);
                                    }

                                    found
                                }
                                _ => false,
                            })
                        {
                            *a = p_idx;
                            debugln!("replaced with {:?}", (p, (a, b, c)));
                        }
                    }

                    break 'site;
                }

                // 9. Create bisector Bpq*.
                let bpq = Bisector::new(sites, p_idx, q_idx);

                // 10. Update list L so that it contains ..., Rq*, Cpq-, Rp*, Cpq+, Rq*, ... in
                //     place of Rq*.
                benchline.split3(reg_q_idx, (bpq.c_minus(sites), p_idx, bpq.c_plus(sites)));

                // 11. Delete from Q the intersection between the left and right boundary of Rq*,
                //     if any.
                'middle: {
                    let Some(left_neighbor) = benchline.get_left_boundary(reg_q_idx) else {
                        break 'middle;
                    };

                    let Some(right_neighbor) = benchline.get_right_boundary(reg_q_idx + 2) else {
                        break 'middle;
                    };

                    events.remove(|e| match e {
                        &Event::Intersection(_, (a, b, c)) => {
                            let retain = (a == left_neighbor.a || a == left_neighbor.b)
                                && (b == right_neighbor.a || b == right_neighbor.b)
                                && (c == right_neighbor.a || c == right_neighbor.b);

                            if retain {
                                debugln!("removing {:?}", e);
                            }

                            retain
                        }
                        _ => false,
                    });
                }

                // 12. Insert into Q the intersection between Cpq- and its neighbor to the left on
                //     L, if any, and the intersection between Cpq+, and its neighbor to the right,
                //     if any.

                'left: {
                    let Some(left_neighbor) = benchline.get_left_boundary(reg_q_idx) else {
                        break 'left;
                    };

                    debugln!("cpq- {:?}", bpq.c_minus(sites));
                    debugln!("left neighbor {:?}", left_neighbor);

                    let Some(p) = bpq.c_minus(sites).star_intersection(sites, left_neighbor) else {
                        break 'left;
                    };

                    let q = benchline.get_region(reg_q_idx - 1);
                    let r = benchline.get_region(reg_q_idx);
                    let s = benchline.get_region(reg_q_idx + 1);
                    events.push(Event::Intersection(p, (q, r, s)));
                    debugln!("l intersection {:?}, ({:?}, {:?}, {:?})", p, q, r, s);
                }

                'right: {
                    let Some(right_neighbor) = benchline.get_right_boundary(reg_q_idx + 2) else {
                        break 'right;
                    };

                    debugln!("cpq+ {:?}", bpq.c_plus(sites));
                    debugln!("right neighbor {:?}", right_neighbor);

                    let Some(p) = bpq.c_plus(sites).star_intersection(sites, right_neighbor) else {
                        break 'right;
                    };

                    let q = benchline.get_region(reg_q_idx + 1);
                    let r = benchline.get_region(reg_q_idx + 2);
                    let s = benchline.get_region(reg_q_idx + 3);
                    events.push(Event::Intersection(p, (q, r, s)));
                    debugln!("r intersection {:?}, ({:?}, {:?}, {:?})", p, q, r, s);
                }
            }
            // 13. p is an intersection:
            Event::Intersection(p, (q_idx, r_idx, s_idx)) => {
                let q = sites[q_idx as usize];
                let r = sites[r_idx as usize];
                let s = sites[s_idx as usize];

                // 14. Let p be the intersection of boundaries Cqr and Crs.

                // 15. Create the bisector Bqs*.
                // 16. Update list L so it contains Cqs = Cqs- or Cqs+,as appropriate, instead of Cqr, R*r, Crs.

                let region_r_idx = benchline.find_region3(q_idx, r_idx, s_idx).unwrap();

                let cqs = Bisector::c_merge(sites, p, q_idx, r_idx, s_idx);
                let cqr = benchline.get_left_boundary(region_r_idx).unwrap();

                let crs = benchline.get_right_boundary(region_r_idx).unwrap();

                benchline.merge(region_r_idx, cqs);

                // 17. Delete from Q any intersection between Cqr and its neighbor to the left and between Crs and its neighbor to the right.
                'left: {
                    let Some(left_neighbor) = benchline.get_left_boundary(region_r_idx - 1) else {
                        break 'left;
                    };

                    debugln!("cqr {:?}", cqr);
                    debugln!("left neighbor {:?}", left_neighbor);

                    events.remove(|e| match e {
                        &Event::Intersection(_, (a, b, c)) => {
                            let retain = (a == left_neighbor.a || a == left_neighbor.b)
                                && (b == cqr.a || b == cqr.b)
                                && (c == cqr.a || c == cqr.b);

                            if retain {
                                debugln!("removing {:?}", e);
                            }

                            retain
                        }
                        _ => false,
                    });
                }

                'right: {
                    let Some(right_neighbor) = benchline.get_right_boundary(region_r_idx) else {
                        break 'right;
                    };

                    debugln!("crs {:?}", crs);
                    debugln!("right neighbor {:?}", right_neighbor);

                    events.remove(|e| match e {
                        &Event::Intersection(_, (a, b, c)) => {
                            let retain = (a == crs.a || a == crs.b)
                                && (b == crs.a || b == crs.b)
                                && (c == right_neighbor.a || c == right_neighbor.b);

                            if retain {
                                debugln!("removing {:?}", e);
                            }

                            retain
                        }
                        _ => false,
                    });
                }

                // 18. Insert any intersections between Cqs and its neighbors to the left or right into Q.

                let cqs = if cqs.max_x.is_finite() {
                    // a bisector going left should intersect a coincident bisector.
                    Bisector {
                        max_x: f32_next_up(cqs.max_x),
                        ..cqs
                    }
                } else {
                    cqs
                };

                debugln!("cqs {:?}", cqs);

                'left: {
                    let Some(left_neighbor) = benchline.get_left_boundary(region_r_idx - 1) else {
                        break 'left;
                    };

                    debugln!("left neighbor {:?}", left_neighbor);

                    let Some(i) = cqs.star_intersection(sites, left_neighbor) else {
                        break 'left;
                    };

                    // if a new intersection happens at the same point, it came from a boundary
                    // that passes through this point. In that case, we consider that boundary to
                    // be to the left of this one, so we should ignore this intersection.
                    if i == p && p == sites[cqs.a as usize] {
                        break 'left;
                    }

                    let q = benchline.get_region(region_r_idx - 2);
                    let r = benchline.get_region(region_r_idx - 1);
                    let s = benchline.get_region(region_r_idx);
                    events.push(Event::Intersection(i, (q, r, s)));
                    debugln!("il intersection {:?}, ({:?}, {:?}, {:?})", i, q, r, s);
                }

                'right: {
                    let Some(mut right_neighbor) = benchline.get_right_boundary(region_r_idx)
                    else {
                        break 'right;
                    };

                    debugln!("right neighbor {:?}", right_neighbor);

                    // We should be able to intersect with the right neighbor, even if it start at
                    // this point. This is need to handle coincident intersections.
                    if right_neighbor.max_x.is_finite() {
                        right_neighbor.max_x = f32_next_up(right_neighbor.max_x);
                    }

                    let Some(p) = cqs.star_intersection(sites, right_neighbor) else {
                        break 'right;
                    };

                    let q = benchline.get_region(region_r_idx - 1);
                    let r = benchline.get_region(region_r_idx);
                    let s = benchline.get_region(region_r_idx + 1);
                    events.push(Event::Intersection(p, (q, r, s)));
                    debugln!("ir intersection {:?}, ({:?}, {:?}, {:?})", p, q, r, s);
                }

                // 19. Mark p as a vertex and as an endpoint of Bqr*, Brs*, and Bqs*.
                let p_unstar = circumcenter(q, r, s).unwrap();
                debugln!("circuncenter of {:?} {:?} {:?}: {:?}", q, r, s, p_unstar);
                debugln!("vertex {} {} {}: {:?}", q_idx, r_idx, s_idx, p_unstar);
                cells[q_idx as usize].add_vertex(sites, p_unstar, q_idx, r_idx, s_idx);
                cells[r_idx as usize].add_vertex(sites, p_unstar, r_idx, q_idx, s_idx);
                cells[s_idx as usize].add_vertex(sites, p_unstar, s_idx, q_idx, r_idx);
            }
        }
        on_progress(&benchline, events.as_slice(), &cells);
    }

    // 20. end
    cells
}

/// Compare the polar angle of the vectors `a` and `b` in relation to the x-axis, in the  range [0,
/// τ) Return if the polar angle of `a` is less, equal or greater than of `b`.
///
/// If either `a` or `b` is a zero vector, this will return `Ordering::Equal`.
///
/// Based on: https://stackoverflow.com/a/39420680
pub fn vec2_angle_cmp(a: Point, b: Point) -> Ordering {
    let h_a = (a.y < 0.0) || ((a.y == 0.0) && (a.x < 0.0)); // 0 for [0,180). 1 for [180,360).
    let h_b = (b.y < 0.0) || ((b.y == 0.0) && (b.x < 0.0)); // 0 for [0,180). 1 for [180,360).

    if h_a == h_b {
        // bxa = |b|.|a|.sin(angle from `b` to `a`, positive in orientation `+x` to `+y`)
        let bxa = b.x * a.y - b.y * a.x;
        return bxa.partial_cmp(&0.0).unwrap_or(Ordering::Equal);
    }

    if h_a {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

// A cell of the voronoi diagram. Contains a list of neighbors and points.
//
// The mediatriz of this site with the neighbors define the edges of the voronoi cell. The points
// are where the edges intersect. If two neighbor mediatrizes don't intersect, its corresponding
// intersection point will be Point::NAN.
#[derive(Clone, Debug)]
pub struct Cell {
    /// Neighbors, sorted by they positive angle in relation to the x-axis, with x->y orientation.
    pub neighbors: Vec<SiteIdx>,
    /// The points where the edges of the cell intersect.
    ///
    /// The nth-point is the intersection of the mediatriz of the nth-neighbor with the mediatriz
    /// of the (n+1)th-neighbor. n+1 wraps around.
    pub points: Vec<Point>,
}
impl Cell {
    fn new() -> Self {
        Cell {
            neighbors: Vec::new(),
            points: Vec::new(),
        }
    }

    /// Add neighbor. Return the index where the neighbor was inserted.
    fn add_neighbor(&mut self, sites: &[Point], this_idx: SiteIdx, neighbor: SiteIdx) -> usize {
        debugln!(
            "{}: adding {:?} to {:?} ({:?})",
            this_idx,
            neighbor,
            self.neighbors,
            self.points
        );

        for i in 0..self.neighbors.len() {
            if neighbor == self.neighbors[i] {
                return i;
            }

            let b_idx = self.neighbors[i];
            let ord = angle_cmp(sites, this_idx, neighbor, b_idx);

            if ord.is_lt() {
                self.neighbors.insert(i, neighbor);
                self.points.insert(i, Point::NAN);
                return i;
            }
        }

        self.neighbors.push(neighbor);
        self.points.push(Point::NAN);
        self.neighbors.len() - 1
    }

    /// Add the intersection point of the two bisectors with the two given neighbors.
    ///
    /// The point is inserted in the correct position in the list of points, and a placeholder is
    /// added if necessary to keep the constraint `points.len == neighbors.len-1`.
    fn add_vertex(
        &mut self,
        sites: &[Point],
        point: Point,
        this_idx: SiteIdx,
        mut a_idx: SiteIdx,
        mut b_idx: SiteIdx,
    ) {
        debug_assert!(!point.x.is_nan());
        debug_assert!(!point.y.is_nan());
        // PERF: a and b should be consecutive neighbors, so after we find one we could insert the
        // other with a single extra comparison.

        let ord = {
            let mut db = sites[b_idx as usize] - sites[this_idx as usize];
            if db == Point::new(0.0, 0.0) {
                if b_idx < this_idx {
                    db.x = -1.0;
                } else {
                    db.x = 1.0;
                };
            }
            let mut da = sites[a_idx as usize] - sites[this_idx as usize];
            if da == Point::new(0.0, 0.0) {
                if a_idx < this_idx {
                    da.x = -1.0;
                } else {
                    da.x = 1.0;
                };
            }

            let cross = da.x * db.y - da.y * db.x;
            cross.partial_cmp(&0.0).unwrap_or(Ordering::Equal).reverse()
        };

        if ord.is_gt() {
            std::mem::swap(&mut a_idx, &mut b_idx);
        }

        debugln!(
            "{}: adding {:?} {:?} to {:?} ({:?}): {:?}",
            this_idx,
            a_idx,
            b_idx,
            self.neighbors,
            self.points,
            point
        );

        self.add_neighbor(sites, this_idx, b_idx);
        let i = self.add_neighbor(sites, this_idx, a_idx);

        debug_assert!(self.points[i].is_nan());
        self.points[i] = point;

        debugln!(
            "{}: added  {:?} {:?} to {:?} ({:?}): {:?}",
            this_idx,
            a_idx,
            b_idx,
            self.neighbors,
            self.points,
            point
        );
    }

    #[cfg(test)]
    /// Check if the cell is convex. It should be.
    fn is_convex(&self) -> bool {
        let n = self.neighbors.len();
        if n < 3 {
            return true;
        }

        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];
            let c = self.points[(i + 2) % n];

            let ab = b - a;
            let bc = c - b;

            // ab x bc = sin(θ) * |ab| * |bc|
            // (θ >= 0) <-> (ab x bc >= 0)
            let cross = ab.x * bc.y - ab.y * bc.x;
            let dot = ab.x * bc.x + ab.y * bc.y;

            debugln!(
                "{}: cross {} = {:?} x {:?} ({})",
                i,
                cross,
                b - a,
                c - b,
                dot
            );

            // If the cell is not closed, some of the points will be NaN. Ignore them.
            if cross.is_nan() {
                continue;
            }

            if cross < -dot.abs() * 1e-3 {
                return false;
            }
        }

        true
    }
}

fn angle_cmp(sites: &[Point], this_idx: u32, neighbor: u32, b_idx: u32) -> Ordering {
    let mut db = sites[b_idx as usize] - sites[this_idx as usize];
    if db == Point::new(0.0, 0.0) {
        if b_idx < this_idx {
            db.x = -1.0;
        } else {
            db.x = 1.0;
        };
    }
    let mut da = sites[neighbor as usize] - sites[this_idx as usize];
    if da == Point::new(0.0, 0.0) {
        if neighbor < this_idx {
            da.x = -1.0;
        } else {
            da.x = 1.0;
        };
    }

    vec2_angle_cmp(da, db).then_with(|| {
        // if they have the same direction, the one with smaller index is to the left.
        let ord = neighbor.cmp(&b_idx);
        if db.y > 0.0 {
            ord.reverse()
        } else {
            ord
        }
    })
}

/// A segment of the bisector of two sites.
#[derive(Clone, Copy)]
pub struct Bisector {
    /// The higher point, the minimun point of the hyperbola "bisector*".
    pub a: SiteIdx,
    /// The lower point.
    pub b: SiteIdx,
    /// The x value of the leftmost point of the hyperbola segment, inclusive.
    pub min_x: f32,
    /// The x value of the rightmost point of the hyperbola segment, exclusive.
    pub max_x: f32,
}

impl std::fmt::Debug for Bisector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Bisector")
            .field(&self.a)
            .field(&self.b)
            .field(&(self.min_x..self.max_x))
            .finish()
    }
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

    pub fn new(sites: &[Point], mut a: SiteIdx, mut b: SiteIdx) -> Self {
        let (p, q) = (sites[a as usize], sites[b as usize]);

        // PERF: maybe the caller can ensure this is true?
        if p < q {
            std::mem::swap(&mut a, &mut b);
        }

        // If py = qy, then bisector* is a vertical half-line.
        if p.y == q.y {
            let mx = (p.x + q.x) / 2.0;

            assert!(f32_next_up(mx) > mx);

            return Self {
                a,
                b,
                min_x: mx,
                max_x: f32_next_up(mx),
            };
        }

        Self {
            a,
            b,
            min_x: f32::NEG_INFINITY,
            max_x: f32::INFINITY,
        }
    }

    // Cpq- to the left o f and containing p.
    fn c_minus(self, sites: &[Point]) -> Bisector {
        let (a, b) = self.ab(sites);

        // If py = qy, then we set Cpq- = {} and Cpq+ = *p(Bpq).
        if a.y == b.y {
            return Self {
                min_x: f32::INFINITY,
                max_x: f32::NEG_INFINITY,
                ..self
            };
        }

        Bisector { max_x: a.x, ..self }
    }

    // Cpq+ to the right o f and containing p.
    fn c_plus(self, sites: &[Point]) -> Bisector {
        let (a, b) = self.ab(sites);

        // If py = qy, then we set Cpq- = {} and Cpq+ = *p(Bpq).
        if a.y == b.y {
            return self;
        }

        Bisector { min_x: a.x, ..self }
    }

    /// The y value of the line bisector at x.
    pub fn y_at(&self, sites: &[Point], x: f32) -> f32 {
        let (a, b) = self.ab(sites);
        line_equation(x, a.x, a.y, b.x, b.y)
    }

    /// The y value of the hyperbola "bisector*" at x.
    pub fn y_star_at(&self, sites: &[Point], x: f32) -> f32 {
        // let (a, b) = self.ab(sites);
        //
        // // It is important to guarantee this, when leading with coincident points.
        // if x == a.x {
        //     return a.y;
        // }
        //
        // let dx = b.x - a.x;
        // let dy = b.y - a.y;
        // let x = x - a.x;
        //
        // let x2 = x * x;
        // let dx2 = dx * dx;
        // let dy2 = dy * dy;
        //
        // let i1 = (dx2 - 2.0 * dx * x + dy2) / (2.0 * dy);
        // a.y + (x2 + (i1).powi(2)).sqrt() + i1

        let y = self.y_at(sites, x);
        y + dist(sites[self.a as usize], Point::new(x, y))
    }

    /// Return if point is on the left side or right side of the hyperbola, obtained by the
    /// *-mapping of the bisector.
    fn star_cmp(&self, sites: &[Point], point: Point) -> Ordering {
        let (a, b) = self.ab(sites);

        if point.x < self.min_x {
            debugln!("less! {} {}", point.x, self.min_x);
            return Ordering::Less;
        }
        if point.x >= self.max_x {
            debugln!("great! {} {}", point.x, self.max_x);
            return Ordering::Greater;
        }

        if (b.y - a.y) == 0.0 {
            debugln!("equal! {} {}", point.x, a.x);
            // the bisector is a vertical line segment
            return point
                .x
                .partial_cmp(&a.x)
                .unwrap()
                // if we are on the vertical line, we are on the left side (as the logic below),
                // unless we are at the boundary of a coincident point, them we are on the right
                // side (because this point index is greater).
                .then_with(|| {
                    if point == a {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                });
        }

        // Calculates the intersection point of the bisector* with the horizontal line that pass
        // through `point`. Then check if it is to the left or to the right.
        let bisector_star_x = self.x_at_y_star(sites, point.y);

        debugln!("{} <=> {} ({})", point.x, bisector_star_x, a.x);

        let ord = point.x.partial_cmp(&bisector_star_x).unwrap();

        // let bisector_star_y = self.y_star_at(sites, point.x);
        //
        // debugln!("{} <=> {}", point.y, bisector_star_y);
        //
        // let ord = point.y.partial_cmp(&bisector_star_y).unwrap();
        //
        // // if this is the right half of the hyperbola, above y means left side.
        // let ord = if self.min_x >= a.x {
        //     ord.reverse()
        // } else {
        //     ord
        // };

        // if a point `q` is on the bisector, it will be on the left side, in order to Cq_+ to
        // intersect with the bisector (Cqr- would not intersect it, because it don't contain `q`
        // in its domain).
        ord.then(Ordering::Less)
    }

    /// Calculates the intersection point of the bisector* with the horizontal line at the given y
    /// coordinate.
    pub fn x_at_y_star(&self, sites: &[Point], y: f32) -> f32 {
        let (a, b) = self.ab(sites);

        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let y = y - a.y;

        if dy == 0.0 {
            // vertical boundary
            return (a.x + b.x) / 2.0;
        }

        let dx2 = dx * dx;
        let dy2 = dy * dy;

        let t1 = (y * (-dy + y)).sqrt();
        let t2 = (dx2 + dy2).sqrt();

        // If we are on `cqs-` or `cqs+`, we choose the corresponding solution.
        if (self.min_x.is_finite()) == (dy > 0.0) {
            a.x + (dx * y + t1 * t2) / dy
        } else {
            a.x + (dx * y - t1 * t2) / dy
        }
    }

    /// Calculates the intersection point of the bisector with the horizontal line at the given y.
    pub fn x_at_y(&self, sites: &[Point], y: f32) -> f32 {
        let (a, b) = self.ab(sites);
        line_equation(y, a.y, a.x, b.y, b.x)
    }

    /// Returns the intersection point of two bisectors.
    fn intersection(&self, sites: &[Point], other: Bisector) -> Option<Point> {
        let (a, b) = self.ab(sites);
        let (oa, ob) = other.ab(sites);

        if self.min_x >= other.max_x || self.max_x <= other.min_x {
            debugln!("out of domain");
            return None;
        }

        let px = a.x;
        let py = a.y;
        let qx = b.x;
        let qy = b.y;

        // py=qy: vertical boundary
        if py == qy {
            debugln!("vertical 1!");
            let x = (px + qx) / 2.0;
            let y = other.y_at(sites, x);
            return Some(Point::new(x, y));
        }

        let rx = oa.x;
        let ry = oa.y;
        let sx = ob.x;
        let sy = ob.y;

        // ry=sy: vertical boundary
        if ry == sy {
            debugln!("vertical 2!");
            let x = (rx + sx) / 2.0;
            let y = self.y_at(sites, x);
            return Some(Point::new(x, y));
        }

        let px2 = px * px;
        let py2 = py * py;
        let qx2 = qx * qx;
        let qy2 = qy * qy;
        let rx2 = rx * rx;
        let ry2 = ry * ry;
        let sx2 = sx * sx;
        let sy2 = sy * sy;

        let d =
            2.0 * (px * ry - px * sy - py * rx + py * sx - qx * ry + qx * sy + qy * rx - qy * sx);

        if d == 0.0 {
            debugln!("divided by zero!");
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

        let mut x = nx / d;
        let y = ny / d;

        // If the intersection lies outside the domain of `other`, but `self` has origin in a point
        // to the right of `other`, and it inside the domain, because that intersection should
        // happen.
        if self.max_x.is_finite()
            && x == self.max_x
            && other.star_cmp(sites, sites[self.a as usize]) == Ordering::Greater
        {
            x = f32_next_down(x);
        }

        if x < self.min_x || x >= self.max_x || x < other.min_x || x >= other.max_x {
            debugln!(
                "out of domain 2, {} {} {} {} {}",
                x,
                self.min_x,
                self.max_x,
                other.min_x,
                other.max_x
            );
            return None;
        }

        let p = Point::new(x, y);
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

    /// Return the bisector created at the intersection of the two bisectors (step 16 of the
    /// algoritmh). Returns Cqs+, Cqs- or Cqs0, as appropriated.
    ///
    /// From paper, "Cqs is Cqs+ either if p is to the right of the higher of q and s or if q
    /// and s are cohorizontal; otherwise Cqs is Cqs-.". Modified to better handle vertical
    /// bisectors (Cqs0).
    fn c_merge(sites: &[Point], p: Point, q_idx: u32, _r_idx: u32, s_idx: u32) -> Bisector {
        let q = sites[q_idx as usize];
        let s = sites[s_idx as usize];
        let bqs = Bisector::new(sites, q_idx, s_idx);

        if q.y == s.y {
            return bqs; // cqs0
        }

        let plus = p.x >= q.max(s).x;
        if plus {
            // cqs+
            Bisector { min_x: p.x, ..bqs }
        } else {
            // cqs-
            Bisector { max_x: p.x, ..bqs }
        }
    }
}

fn line_equation(x: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let x = x - ax;
    let dx = bx - ax;
    let dy = by - ay;

    let dx2 = dx * dx;
    let dy2 = dy * dy;

    ay + (dx2 - 2.0 * dx * x + dy2) / (2.0 * dy)
}

fn dist(a: Point, b: Point) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

/// Returns the *-mapping of the point `p` with respect to the region of site `q`.
///
/// *(x,y) = (x, y + dist(q, (x, y))
fn start_map(p: Point, q: Point) -> Point {
    let x = p.x;
    let y = p.y;
    Point::new(x, y + dist(q, p))
}

/// Finds the circumcenter of the triangle formed by the points `a`, `b`, and `c`.
fn circumcenter(a: Point, b: Point, c: Point) -> Option<Point> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

    if d == 0.0 {
        // assume the two points that are coincident are horizotally side by side.
        let (a, b) = if a == b {
            (a, c)
        } else if a == c {
            (a, b)
        } else if b == c {
            (c, a)
        } else {
            return None;
        };

        let y = line_equation(a.x, a.x, a.y, b.x, b.y);

        return Some(Point::new(a.x, y));
    }

    let ux = (a.x * a.x + a.y * a.y) * (b.y - c.y)
        + (b.x * b.x + b.y * b.y) * (c.y - a.y)
        + (c.x * c.x + c.y * c.y) * (a.y - b.y);
    let uy = (a.x * a.x + a.y * a.y) * (c.x - b.x)
        + (b.x * b.x + b.y * b.y) * (a.x - c.x)
        + (c.x * c.x + c.y * c.y) * (b.x - a.x);

    Some(Point::new(ux / d, uy / d))
}

/// From https://github.com/rust-lang/rust/pull/100578
fn f32_next_up(x: f32) -> f32 {
    // We must use strictly integer arithmetic to prevent denormals from
    // flushing to zero after an arithmetic operation on some platforms.
    const TINY_BITS: u32 = 0x1; // Smallest positive f32.
    const CLEAR_SIGN_MASK: u32 = 0x7fff_ffff;

    let bits = x.to_bits();
    if x.is_nan() || bits == f32::INFINITY.to_bits() {
        return x;
    }

    let abs = bits & CLEAR_SIGN_MASK;
    let next_bits = if abs == 0 {
        TINY_BITS
    } else if bits == abs {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next_bits)
}

/// From https://github.com/rust-lang/rust/pull/100578
fn f32_next_down(x: f32) -> f32 {
    // We must use strictly integer arithmetic to prevent denormals from
    // flushing to zero after an arithmetic operation on some platforms.
    const NEG_TINY_BITS: u32 = 0x8000_0001; // Smallest (in magnitude) negative f32.
    const CLEAR_SIGN_MASK: u32 = 0x7fff_ffff;

    let bits = x.to_bits();
    if x.is_nan() || bits == f32::NEG_INFINITY.to_bits() {
        return x;
    }

    let abs = bits & CLEAR_SIGN_MASK;
    let next_bits = if abs == 0 {
        NEG_TINY_BITS
    } else if bits == abs {
        bits - 1
    } else {
        bits + 1
    };
    f32::from_bits(next_bits)
}
