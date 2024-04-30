use std::ops::Not;

use super::{circumcenter, dist, fortune_algorithm, vec2, Bisector, Cell, Point, SiteIdx};
use proptest::prelude::*;
use proptest::test_runner::TestRunner;
use rand::{seq::SliceRandom, Rng};

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < (a + b).abs() * 2e-5
}

#[test]
fn bisector_intersection() {
    let sites = &[
        Point { pos: vec2(0., 0.) },
        Point { pos: vec2(1., 0.) },
        Point { pos: vec2(0., 1.) },
        Point { pos: vec2(1., 1.) },
    ];

    let ad = super::Bisector::new(sites, 0, 3);
    let bc = super::Bisector::new(sites, 1, 2);

    let p = ad.intersection(sites, bc).unwrap();

    assert_eq!(p.pos, vec2(0.5, 0.5));
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

fn bisector_intersections_(a: (u8, u8), b: (u8, u8), c: (u8, u8), d: (u8, u8)) -> bool {
    let a = Point {
        pos: vec2(a.0 as f32, a.1 as f32),
    };
    let b = Point {
        pos: vec2(b.0 as f32, b.1 as f32),
    };
    let c = Point {
        pos: vec2(c.0 as f32, c.1 as f32),
    };
    let d = Point {
        pos: vec2(d.0 as f32, d.1 as f32),
    };

    let sites = &[a, b, c, d];
    let ab = super::Bisector::new(sites, 0, 1);
    let cd = super::Bisector::new(sites, 2, 3);

    println!("ab: {:?}, cd: {:?}", ab, cd);

    let Some(p) = ab.intersection(sites, cd) else {
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
    let p = Point { pos: vec2(4., 0.) };
    let q = Point { pos: vec2(0., 1.) };
    let r = Point { pos: vec2(5., 2.) };

    let sites = &[p, q, r];

    let bpq = Bisector::new(sites, 0, 1).c_plus(sites);

    let y = bpq.y_star_at(sites, r.pos.x);

    println!("y: {}", y);

    assert_eq!(bpq.star_cmp(sites, r), std::cmp::Ordering::Greater);
}

#[test]
fn bisector_no_intersection() {
    let p = Point { pos: vec2(1., 0.) };
    let q = Point { pos: vec2(0., 1.) };
    let r = Point { pos: vec2(1., 1.) };

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
    let p = Point { pos: vec2(1., 0.) };
    let q = Point { pos: vec2(0., 1.) };
    let r = Point { pos: vec2(1., 1.) };
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
        println!("regions: {:?}", regions);
        assert_eq!(regions, expected_benchline[0]);
        expected_benchline = &expected_benchline[1..];
    });

    for cell in vertexes.into_iter() {
        println!("cell {:?}", cell);

        let points: Vec<Point> = cell
            .points
            .into_iter()
            .filter(|v| v.is_nan().not())
            .collect();

        assert_eq!(cell.neighbors.len(), 2);
        assert_eq!(points.len(), 1);

        println!("points: {:?}", points);

        assert_eq!(points[0], intersection);
    }
}

#[test]
fn diagram_fuzz() {
    let mut runner = TestRunner::default();

    let points = proptest::collection::vec((0..5u8, 0..5u8), 0..4);

    runner
        .run(&points, |points| {
            diagram_fuzz_(points);
            Ok(())
        })
        .unwrap();
}

// #[test]
// fn diagram_fuzz1() {
//     diagram_fuzz_(vec![
//         (151, 151),
//         (0, 44),
//         (9, 41),
//         (37, 8),
//         (0, 23),
//         (99, 0),
//         (0, 45),
//         (0, 46),
//         (99, 142),
//     ])
// }

fn diagram_fuzz_(mut points: Vec<(u8, u8)>) {
    points.sort();
    points.dedup();

    let points = points
        .into_iter()
        .map(|(x, y)| Point {
            pos: vec2(x as f32, y as f32),
        })
        .collect::<Vec<_>>();

    fortune_algorithm(&points, &mut |_, _| {});
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
        .map(|(x, y)| Point {
            pos: vec2(*x as f32, *y as f32),
        })
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
        println!("regions: {:?}", regions);
        if regions != expected_benchline[0] {
            println!("expecte: {:?}", expected_benchline[0]);
        }
        assert_eq!(regions, expected_benchline[0]);
        expected_benchline = &expected_benchline[1..];
    });

    assert!(expected_benchline.is_empty());
    assert_eq!(vertexes.len(), points.len());

    for (cell, &site) in vertexes.into_iter().zip(&points) {
        println!("cell {:?}", cell);

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

            println!("dist_p: {}, dist_neigh: {}", dist_p, dist_neigh);

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
    let p = Point {
        pos: vec2(a.0 as f32, a.1 as f32),
    };
    let q = Point {
        pos: vec2(b.0 as f32, b.1 as f32),
    };

    let sites = &[p, q];

    if p.pos.y == q.pos.y {
        return true;
    }

    let bisector = Bisector::new(sites, 0, 1);

    let mean = (p.pos + q.pos) / 2.0;

    let y_at = bisector.y_at(sites, mean.x);

    println!("a: {:?}, b: {:?}, mean: {}, y_at: {}", a, b, mean, y_at);

    close(y_at, mean.y)
}

#[test]
fn bisector_y_at1() {
    let p = Point {
        pos: vec2(1.0, 1.0),
    };
    let q = Point {
        pos: vec2(0.0, 0.0),
    };

    let sites = &[p, q];

    let bisector = Bisector::new(sites, 0, 1);

    let y_at = bisector.y_at(sites, 0.5);

    println!("y_at: {}", y_at);

    assert!(close(y_at, 0.5))
}

#[test]
fn y_star_at() {
    let p = Point {
        pos: vec2(5.0, 2.0),
    };
    let q = Point {
        pos: vec2(4.0, 0.0),
    };
    let r = Point {
        pos: vec2(5.0, 3.0),
    };

    let sites = &[p, q, r];

    let bpq = Bisector::new(sites, 0, 1);

    let y_star = bpq.y_star_at(sites, r.pos.x);

    println!("y_star: {}", y_star);

    assert!(close(y_star, 2.0));
}

proptest! {
    #[test]
    fn test_circumcenter_(a: (u8, u8), b: (u8, u8), c: (u8, u8)) {
        test_circumcenter(a, b, c);
    }
}

fn test_circumcenter(a: (u8, u8), b: (u8, u8), c: (u8, u8)) {
    // check if points are collinear
    {
        let (ax, ay) = (a.0 as i32, a.1 as i32);
        let (bx, by) = (b.0 as i32, b.1 as i32);
        let (cx, cy) = (c.0 as i32, c.1 as i32);

        if (by - ay) * (cx - bx) == (cy - by) * (bx - ax) {
            return;
        }
    }

    let a = Point {
        pos: vec2(a.0 as f32, a.1 as f32),
    };
    let b = Point {
        pos: vec2(b.0 as f32, b.1 as f32),
    };
    let c = Point {
        pos: vec2(c.0 as f32, c.1 as f32),
    };

    let cc = circumcenter(a, b, c);

    let r1 = dist(cc, a);
    let r2 = dist(cc, b);
    let r3 = dist(cc, c);

    println!("r1: {}, r2: {}, r3: {}", r1, r2, r3);

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
        .map(|&(x, y)| Point {
            pos: vec2(x as f32, y as f32),
        })
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
        cell.add_point(sites, p, 0, a as SiteIdx, b as SiteIdx);
    }

    assert_eq!(cell.points, hull_points);
}
