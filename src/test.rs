use std::ops::Not;

use super::{circumcenter, debugln, dist, fortune_algorithm, vec2, Bisector, Cell, Point, SiteIdx};
use proptest::prelude::*;
use proptest::test_runner::TestRunner;
use rand::{seq::SliceRandom, Rng};

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < (a + b).abs() * 2e-5
}

#[test]
fn bisector_intersection() {
    let sites = &[
        Point::new(0., 0.),
        Point::new(1., 0.),
        Point::new(0., 1.),
        Point::new(1., 1.),
    ];

    let ad = super::Bisector::new(sites, 0, 3);
    let bc = super::Bisector::new(sites, 1, 2);

    let p = ad.intersection(sites, bc).unwrap();

    assert_eq!(p.pos, vec2(0.5, 0.5));
}

#[test]
fn bisector_intersection1() {
    let sites = &[Point::new(0., 0.), Point::new(0., 0.), Point::new(0., 1.)];

    let cpq_plus = super::Bisector::new(sites, 2, 0).c_plus(sites);
    let right_neighbor = super::Bisector::new(sites, 1, 0);

    debugln!("cpq_plus: {:?}", cpq_plus);
    debugln!("right_neighbor: {:?}", right_neighbor);

    let p = cpq_plus.intersection(sites, right_neighbor);

    debugln!("p: {:?}", p);

    assert!(p.is_some());
}

#[test]
fn bisector_intersections() {
    let mut runner = TestRunner::default();
    runner
        .run(
            &(
                (0..10u8, 0..10u8),
                (0..10u8, 0..10u8),
                (0..10u8, 0..10u8),
                (0..10u8, 0..10u8),
            ),
            |(a, b, c, d)| {
                bisector_intersections_(a, b, c, d);
                Ok(())
            },
        )
        .unwrap();
}

fn bisector_intersections_(a: (u8, u8), b: (u8, u8), c: (u8, u8), d: (u8, u8)) {
    let a = Point::new(a.0 as f32, a.1 as f32);
    let b = Point::new(b.0 as f32, b.1 as f32);
    let c = Point::new(c.0 as f32, c.1 as f32);
    let d = Point::new(d.0 as f32, d.1 as f32);

    let sites = &[a, b, c, d];
    let ab = super::Bisector::new(sites, 0, 1);
    let cd = super::Bisector::new(sites, 2, 3);

    debugln!("ab: {:?}, cd: {:?}", ab, cd);

    let Some(p) = ab.intersection(sites, cd) else {
        return;
    };

    debugln!("p: {:?}", p);

    let da = dist(a, p);
    let db = dist(b, p);
    let dc = dist(c, p);
    let dd = dist(d, p);

    // Ignore NaN/Infinite results
    if !da.is_finite() || !db.is_finite() || !dc.is_finite() || !dd.is_finite() {
        return;
    }

    debugln!("da: {}, db: {}, dc: {}, dd: {}", da, db, dc, dd);

    assert!(close(da, db));
    assert!(close(dc, dd));
}

#[test]
fn bisector_cmp() {
    let p = Point::new(1., 0.);
    let q = Point::new(0., 1.);
    let r = Point::new(1., 1.);
    let t = Point::new(-1., 1.);

    let sites = &[p, q, r, t];

    let bpq = Bisector::new(sites, 0, 1);

    assert_eq!(
        bpq.c_plus(sites).star_cmp(sites, r),
        std::cmp::Ordering::Greater
    );
    assert_eq!(
        bpq.c_minus(sites).star_cmp(sites, t),
        std::cmp::Ordering::Less
    );
}

#[test]
fn bisector_cmp1() {
    let p = Point::new(4., 0.);
    let q = Point::new(0., 1.);
    let r = Point::new(5., 2.);

    let sites = &[p, q, r];

    let bpq = Bisector::new(sites, 0, 1).c_plus(sites);

    let y = bpq.y_star_at(sites, r.pos.x);

    debugln!("y: {}", y);

    assert_eq!(bpq.star_cmp(sites, r), std::cmp::Ordering::Greater);
}

#[test]
fn bisector_no_intersection() {
    let p = Point::new(1., 0.);
    let q = Point::new(0., 1.);
    let r = Point::new(1., 1.);

    let sites = &[p, q, r];

    let bpq = Bisector::new(sites, 0, 1);
    let bqr = Bisector::new(sites, 1, 2);
    let bpr = Bisector::new(sites, 0, 2);

    let cqr_plus = bqr.c_plus(sites);
    let cqr_minus = bqr.c_minus(sites);

    let cpq_minus = bpq.c_minus(sites);
    assert_eq!(cpq_minus.intersection(sites, cqr_plus), None);
    assert_eq!(cpq_minus.intersection(sites, cqr_minus), None);

    let cpr_plus = bpr.c_plus(sites);
    assert_eq!(cpq_minus.intersection(sites, cpr_plus), None);
    assert_eq!(cqr_plus.intersection(sites, cpr_plus), None);
}

#[test]
fn diagram() {
    let p = Point::new(1., 0.);
    let q = Point::new(0., 1.);
    let r = Point::new(1., 1.);
    let points = [p, q, r];

    let intersection = Bisector::new(&points, 0, 1)
        .intersection(&points, Bisector::new(&points, 1, 2))
        .unwrap();

    let mut expected_benchline: &[&[SiteIdx]] = &[
        &[0],             // first region
        &[0, 1, 0],       // insert q
        &[0, 1, 0, 2, 0], // insert r
        &[0, 1, 2, 0],    // intersect q and r
    ];

    let vertexes = fortune_algorithm(&points, &mut |benchline, _| {
        let regions = benchline.get_regions().collect::<Vec<_>>();
        debugln!("regions: {:?}", regions);
        assert_eq!(regions, expected_benchline[0]);
        expected_benchline = &expected_benchline[1..];
    });

    for cell in vertexes.into_iter() {
        debugln!("cell {:?}", cell);

        let points: Vec<Point> = cell
            .points
            .into_iter()
            .filter(|v| v.is_nan().not())
            .collect();

        assert_eq!(cell.neighbors.len(), 2);
        assert_eq!(points.len(), 1);

        debugln!("points: {:?}", points);

        assert_eq!(points[0], intersection);
    }
}

#[test]
fn diagram_fuzz() {
    let mut runner = TestRunner::default();

    // all representable integers floats
    // let i = -(1 << 23)..(1 << 23);
    let i = -0..6;
    let points = proptest::collection::vec((i.clone(), i), 0..8);

    runner
        .run(&points, |points| {
            diagram_fuzz_(points);
            Ok(())
        })
        .unwrap();
}

/// Test duplicated points
#[test]
fn diagram_fuzz_dup() {
    let mut runner = TestRunner::default();

    let i = 0..10i32;
    let points = proptest::collection::vec((i.clone(), i), 0..16);

    runner
        .run(&points, |points| {
            let points = points
                .iter()
                .copied()
                .map(|(x, y)| Point::new(x as f32, y as f32))
                .collect();
            diagram_fuzz_points(points);
            Ok(())
        })
        .unwrap();
}

#[test]
fn diagram_fuzz_dup_1() {
    let points = vec![
        Point::new(4.0, 2.0),
        Point::new(0.0, 5.0),
        Point::new(0.0, 5.0),
        Point::new(0.0, 6.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_2() {
    let points = vec![
        Point::new(3.0, 0.0),
        Point::new(4.0, 1.0),
        Point::new(6.0, 2.0),
        Point::new(6.0, 2.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_3() {
    let points = vec![
        Point::new(7.0, 1.0),
        Point::new(9.0, 1.0),
        Point::new(8.0, 2.0),
        Point::new(8.0, 2.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_4() {
    let points = vec![
        Point::new(5.0, 0.0),
        Point::new(3.0, 1.0),
        Point::new(3.0, 1.0),
        Point::new(2.0, 2.0),
        Point::new(8.0, 4.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_5() {
    let points = vec![
        Point::new(90.0, 40.0),
        Point::new(90.0, 40.0),
        Point::new(90.0, 70.0),
        Point::new(90.0, 70.0),
        Point::new(00.0, 80.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_6() {
    let points = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(1.0, 1.0),
        Point::new(1.0, 1.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_7() {
    let points = vec![
        Point::new(2.0, 1.0),
        Point::new(8.0, 6.0),
        Point::new(8.0, 6.0),
        Point::new(8.0, 6.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_8() {
    let points = vec![
        Point::new(0.0, 1.0),
        Point::new(0.0, 1.0),
        Point::new(1.0, 1.0),
        Point::new(1.0, 1.0),
        Point::new(8.0, 1.0),
        Point::new(8.0, 1.0),
        Point::new(1.0, 2.0),
        Point::new(0.0, 4.0),
        Point::new(0.0, 4.0),
        Point::new(7.0, 4.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_9() {
    let points = vec![
        Point::new(2.0, 0.0),
        Point::new(1.0, 1.0),
        Point::new(0.0, 2.0),
        Point::new(0.0, 2.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_10() {
    let points = vec![
        Point::new(0.0, 0.0),
        Point::new(0.0, 0.0),
        Point::new(0.0, 0.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz_dup_11() {
    let points = vec![
        Point::new(0.0, -25.0),
        Point::new(24.0, -19.0),
        Point::new(0.0, 0.0),
        Point::new(29.0, 0.0),
        Point::new(29.0, 0.0),
    ];

    diagram_fuzz_points(points);
}

#[test]
fn diagram_fuzz1() {
    diagram_fuzz_(vec![(2, 0), (0, 1), (4, 1), (2, 2)])
}

#[test]
fn diagram_fuzz2() {
    diagram_fuzz_(vec![(0, 1), (1, 0), (1, 2)])
}

#[test]
fn diagram_fuzz3() {
    diagram_fuzz_(vec![(0, 0), (1, 0), (2, 0), (3, 0)])
}

#[test]
fn diagram_fuzz4() {
    diagram_fuzz_(vec![(0, 0), (12, 4), (16, 4), (14, 8), (11, 9)])
}

#[test]
fn diagram_fuzz5() {
    diagram_fuzz_(vec![(4, 8), (8, 8), (8, 11), (6, 12)])
}

#[test]
fn diagram_fuzz6() {
    diagram_fuzz_(vec![(6, 0), (6, 3), (8, 4), (11, 5)])
}

#[test]
fn diagram_fuzz7() {
    diagram_fuzz_(vec![(0, 0), (1, 0), (0, 2), (3, 3), (3, 4), (0, 5)])
}

#[test]
fn diagram_fuzz8() {
    diagram_fuzz_(vec![
        (4, 2), //
        (5, 2),
        (3, 3),
        (4, 5),
        (0, 6),
        (2, 7),
    ]);
}

#[test]
fn diagram_fuzz9() {
    diagram_fuzz_(vec![(0, 4), (1, 4), (2, 5), (2, 6)]);
}

#[test]
fn diagram_fuzz10() {
    diagram_fuzz_(vec![(9, 0), (3, 1), (9, 1), (1, 2), (4, 3), (3, 4)]);
}

#[test]
fn diagram_fuzz11() {
    diagram_fuzz_(vec![
        (3, 1), //
        (0, 0),
        (2, 0),
        (3, 3),
        (2, 4),
    ]);
}

#[test]
fn diagram_fuzz12() {
    diagram_fuzz_(vec![(8, -16), (31, -9), (20, -2), (0, 0)]);
}

#[test]
fn diagram_fuzz13() {
    diagram_fuzz_(vec![(15, -29), (7, -14), (21, -2), (0, 0)]);
}

fn diagram_fuzz_(points: Vec<(i32, i32)>) {
    let mut sites = points
        .into_iter()
        .map(|(x, y)| Point::new(x as f32, y as f32))
        .collect::<Vec<_>>();

    // remove duplicates
    {
        let mut hash = std::collections::HashSet::new();
        sites.retain(|v| hash.insert(*v));
    }

    diagram_fuzz_points(sites);
}

fn diagram_fuzz_points(sites: Vec<Point>) {
    debugln!("{:?}", sites);

    let vertexes = fortune_algorithm(&sites, &mut |_, _| {});

    for (i, (cell, &site)) in vertexes.into_iter().zip(&sites).enumerate() {
        debugln!("{}: cell {:?}", i, cell);

        if sites.len() > 1 {
            assert!(!cell.points.is_empty());
        }

        let nan_count = cell.points.iter().filter(|v| v.is_nan()).count();

        assert!(nan_count <= 2);
        if nan_count > 1 {
            // this only happens when the cell has 2 parallel bisectors. This means it have only
            // two oposing neighbors, but could be more than 2 if they are coincident. Or 1, if all
            // points are coincident.
            let neighbors = {
                let mut x = cell.neighbors.clone();
                x.dedup_by_key(|idx| sites[*idx as usize]);
                x
            };

            assert!(neighbors.len() >= 1);
            assert!(neighbors.len() <= 2);
        }

        // the distance of the intersection point to the site should be the same as the distance to
        // the neighbors
        for (&p, &neigh) in cell.points.iter().zip(cell.neighbors.iter()) {
            if p.is_nan() {
                continue;
            }

            let dist_p = dist(p, site);
            let dist_neigh = dist(p, sites[neigh as usize]);

            debugln!("dist_p: {}, dist_neigh: {}", dist_p, dist_neigh);

            assert!(close(dist_p, dist_neigh));
        }
    }
}

#[test]
fn diagram1() {
    let points = [
        (4, 0), // p
        (0, 1), // q
        (5, 2), // r
        (5, 3), // s
    ];

    let points = points
        .iter()
        .map(|(x, y)| Point::new(*x as f32, *y as f32))
        .collect::<Vec<_>>();

    let mut expected_benchline: &[&[SiteIdx]] = &[
        &[0],                   // first region
        &[0, 1, 0],             // insert q in p
        &[0, 1, 0, 2, 0],       // insert r in p
        &[0, 1, 0, 2, 3, 2, 0], // insert s in r
        &[0, 1, 2, 3, 2, 0],    // intersect q p r
        &[0, 1, 3, 2, 0],       // intersect q r s
    ];

    let vertexes = fortune_algorithm(&points, &mut |benchline, _| {
        let regions = benchline.get_regions().collect::<Vec<_>>();
        debugln!("regions: {:?}", regions);
        if regions != expected_benchline[0] {
            debugln!("expecte: {:?}", expected_benchline[0]);
        }
        assert_eq!(regions, expected_benchline[0]);
        expected_benchline = &expected_benchline[1..];
    });

    debugln!("expected_benchline: {:?}", expected_benchline);
    assert!(expected_benchline.is_empty());
    assert_eq!(vertexes.len(), points.len());

    for (cell, &site) in vertexes.into_iter().zip(&points) {
        debugln!("cell {:?}", cell);

        let intersection_count = cell.points.iter().filter(|v| v.is_nan().not()).count();

        assert!(!cell.points.is_empty());
        assert!(intersection_count <= cell.points.len());
        assert!(intersection_count >= cell.points.len() - 2);

        // the distance of the intersection point to the site should be the same as the distance to
        // the neighbors
        for (&p, &neigh) in cell.points.iter().zip(cell.neighbors.iter()) {
            if p.is_nan() {
                continue;
            }

            let dist_p = dist(p, site);
            let dist_neigh = dist(p, points[neigh as usize]);

            debugln!("dist_p: {}, dist_neigh: {}", dist_p, dist_neigh);

            assert!(close(dist_p, dist_neigh));
        }
    }
}

proptest! {
    #[test]
    fn bisector_y_at_(a: (u8, u8), b: (u8, u8)) {
        bisector_y_at(a, b);
    }
}

fn bisector_y_at(a: (u8, u8), b: (u8, u8)) -> bool {
    let p = Point::new(a.0 as f32, a.1 as f32);
    let q = Point::new(b.0 as f32, b.1 as f32);

    let sites = &[p, q];

    if p.pos.y == q.pos.y {
        return true;
    }

    let bisector = Bisector::new(sites, 0, 1);

    let mean = (p.pos + q.pos) / 2.0;

    let y_at = bisector.y_at(sites, mean.x);

    debugln!("a: {:?}, b: {:?}, mean: {}, y_at: {}", a, b, mean, y_at);

    close(y_at, mean.y)
}

#[test]
fn bisector_y_at1() {
    let p = Point::new(1.0, 1.0);
    let q = Point::new(0.0, 0.0);

    let sites = &[p, q];

    let bisector = Bisector::new(sites, 0, 1);

    let y_at = bisector.y_at(sites, 0.5);

    debugln!("y_at: {}", y_at);

    assert!(close(y_at, 0.5))
}

#[test]
fn y_star_at() {
    let p = Point::new(5.0, 2.0);
    let q = Point::new(4.0, 0.0);
    let r = Point::new(5.0, 3.0);

    let sites = &[p, q, r];

    let bpq = Bisector::new(sites, 0, 1);

    let y_star = bpq.y_star_at(sites, r.pos.x);

    debugln!("y_star: {}", y_star);

    assert!(close(y_star, 2.0));
}

proptest! {
    #[test]
    fn test_circumcenter_(a: (u8, u8), b: (u8, u8), c: (u8, u8)) {
        test_circumcenter(a, b, c);
    }
}

fn test_circumcenter(a: (u8, u8), b: (u8, u8), c: (u8, u8)) {
    let p = Point::new(a.0 as f32, a.1 as f32);
    let q = Point::new(b.0 as f32, b.1 as f32);
    let r = Point::new(c.0 as f32, c.1 as f32);

    let cc = circumcenter(p, q, r);

    // check if points are collinear
    'collinear: {
        let (ax, ay) = (a.0 as i32, a.1 as i32);
        let (bx, by) = (b.0 as i32, b.1 as i32);
        let (cx, cy) = (c.0 as i32, c.1 as i32);

        // `circumcenter` can handle a single duplicated point
        let mut sites = vec![p, q, r];
        sites.sort();
        sites.dedup();
        if sites.len() == 2 {
            // unless they are horizontally align
            if sites[0].pos.y == sites[1].pos.y {
                assert!(cc.is_none());
                return;
            }
            break 'collinear;
        }

        if (by - ay) * (cx - bx) == (cy - by) * (bx - ax) {
            assert!(cc.is_none());
            return;
        }
    }

    let cc = cc.unwrap();

    let r1 = dist(cc, p);
    let r2 = dist(cc, q);
    let r3 = dist(cc, r);

    debugln!("r1: {}, r2: {}, r3: {}", r1, r2, r3);

    assert!(close(r1, r2));
    assert!(close(r1, r3));
}

#[test]
fn test_cell() {
    let points = [
        (0, 0),
        (9, 0),
        (7, 7),
        (0, 9),
        (-7, 7),
        (-9, 0),
        (-7, -7),
        (0, -9),
        (7, -7),
    ];

    let sites = &points
        .iter()
        .map(|&(x, y)| Point::new(x as f32, y as f32))
        .collect::<Vec<_>>();

    let mut intersections: Vec<_> = (1..sites.len() as u32)
        .map(|a| {
            let b = a % (sites.len() as u32 - 1) + 1;

            let boa = Bisector::new(sites, 0, a);
            let bob = Bisector::new(sites, 0, b);

            let intersection = boa.intersection(sites, bob).unwrap();

            (a, intersection, b)
        })
        .collect();

    let hull_points = intersections.iter().map(|(_, p, _)| *p).collect::<Vec<_>>();

    let mut cell = Cell::new();

    let mut rng = rand::thread_rng();

    intersections.shuffle(&mut rng);

    for (mut a, p, mut b) in intersections {
        if rng.gen() {
            std::mem::swap(&mut a, &mut b);
        }
        cell.add_vertex(sites, p, 0, a as SiteIdx, b as SiteIdx);
    }

    assert_eq!(cell.points, hull_points);
}

#[test]
fn angle() {
    use std::f32::consts::TAU;

    let sites = vec![
        Point::new(0.0, 0.0),  // 0
        Point::new(0.0, 0.0),  // 1
        Point::new(1.0, 0.0),  // 2
        Point::new(0.0, 1.0),  // 3
        Point::new(-1.0, 0.0), // 4
        Point::new(0.0, -1.0), // 5
    ];

    let a1 = super::angle(&sites, 0, 1);
    let a2 = super::angle(&sites, 1, 0);
    let a3 = super::angle(&sites, 0, 2);
    let a4 = super::angle(&sites, 0, 3);
    let a5 = super::angle(&sites, 0, 4);
    let a6 = super::angle(&sites, 0, 5);

    debugln!("a1: {}", a1);
    debugln!("a2: {}", a2);
    debugln!("a3: {}", a3);
    debugln!("a4: {}", a4);
    debugln!("a5: {}", a5);
    debugln!("a6: {}", a6);

    assert_eq!(a1, 0.0);
    assert_eq!(a2, TAU / 2.0);
    assert_eq!(a3, 0.0);
    assert_eq!(a4, TAU / 4.0);
    assert_eq!(a5, TAU / 2.0);
    assert_eq!(a6, TAU * 3.0 / 4.0);
}

#[test]
fn angle_cmp() {
    use std::f32::consts::TAU;

    let sites = vec![
        Point::new(0.0, 0.0),  // 0
        Point::new(0.0, 1.0),  // 1
        Point::new(0.0, 1.0),  // 2
        Point::new(0.0, -1.0), // 3
        Point::new(0.0, -1.0), // 4
        Point::new(1.0, 0.0),  // 5
    ];

    let a1 = super::angle_cmp(&sites, 0, 1, 2);
    let a2 = super::angle_cmp(&sites, 0, 2, 1);
    let a3 = super::angle_cmp(&sites, 0, 3, 4);
    let a4 = super::angle_cmp(&sites, 0, 4, 3);
    let a5 = super::angle_cmp(&sites, 0, 5, 1);
    let a6 = super::angle_cmp(&sites, 0, 5, 3);

    debugln!("a1: {:?}", a1);
    debugln!("a2: {:?}", a2);
    debugln!("a3: {:?}", a3);
    debugln!("a4: {:?}", a4);
    debugln!("a5: {:?}", a5);
    debugln!("a6: {:?}", a6);

    assert!(a1 == 0.0 && a1.is_sign_negative());
    assert!(a2 == 0.0 && a2.is_sign_positive());
    assert!(a3 == 0.0 && a3.is_sign_positive());
    assert!(a4 == 0.0 && a4.is_sign_negative());
    assert_eq!(a5, TAU / 4.0);
    assert!(close(a6, -TAU / 4.0));
}
