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
    let sites = &[
        Point { pos: vec2(0., 0.) },
        Point { pos: vec2(1., 0.) },
        Point { pos: vec2(0., 1.) },
        Point { pos: vec2(1., 1.) },
    ];

    let ad = super::Bisector::new(0, 3);
    let bc = super::Bisector::new(1, 2);

    let p = ad.intersection(sites, bc).unwrap();

    assert_eq!(p.pos, vec2(0.5, 0.5));
}

#[quickcheck]
fn bisector_intersections(a: Point, b: Point, c: Point, d: Point) -> bool {
    let sites = &[a, b, c, d];
    let ab = super::Bisector::new(0, 1);
    let cd = super::Bisector::new(2, 3);

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

    let bpq = Bisector::new(0, 1);

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

    let bpq = Bisector::new(0, 1).c_plus(sites);

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

    let bpq = Bisector::new(0, 1);
    let bqr = Bisector::new(1, 2);
    let bpr = Bisector::new(0, 2);

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

    let mut expected_benchline: &[&[SiteIdx]] = &[
        &[0],             // first region
        &[0, 1, 0],       // insert q
        &[0, 1, 0, 2, 0], // insert r
        &[0, 1, 2, 0],    // intersect q and r
    ];

    let _vertexes = fortune_algorithm(&points, &mut |benchline, _| {
        let regions = benchline.get_regions().collect::<Vec<_>>();
        println!("regions: {:?}", regions);
        assert_eq!(regions, expected_benchline[0]);
        expected_benchline = &expected_benchline[1..];
    });

    // for (site, vertexes) in vertexes {
    //     println!("site: {:?}", site);
    //     for vertex in &vertexes {
    //         println!("  vertex: {:?}", vertex);
    //     }
    //
    //     let dists: Vec<f32> = vertexes.iter().map(|v| dist(site, *v)).collect();
    //
    //     assert!(dists.iter().all(|d| *d > 0.0));
    //
    //     // all dists should be the same
    //     assert!(dists.windows(2).all(|w| close(w[0], w[1])));
    // }
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

    let _vertexes = fortune_algorithm(&points, &mut |benchline, _| {
        let regions = benchline.get_regions().collect::<Vec<_>>();
        println!("regions: {:?}", regions);
        if regions != expected_benchline[0] {
            println!("expecte: {:?}", expected_benchline[0]);
        }
        assert_eq!(regions, expected_benchline[0]);
        expected_benchline = &expected_benchline[1..];
    });
}

#[quickcheck]
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

    let bisector = Bisector::new(0, 1);

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

    let bisector = Bisector::new(0, 1);

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

    let bpq = Bisector::new(0, 1);

    let y_star = bpq.y_star_at(sites, r.pos.x);

    println!("y_star: {}", y_star);

    assert!(close(y_star, 2.0));
}

#[quickcheck]
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
