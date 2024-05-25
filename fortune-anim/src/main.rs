use macroquad::prelude::*;

use voronoi_sweep::{fortune_algorithm, Bisector, Cell, Event, Point, SiteIdx};

/// I put the the proc-macro in an wrapper function to workaround it breaking rust-analyzer quick
/// actions.
#[macroquad::main(window_conf)]
async fn main() {
    main_().await
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Voronoi".to_owned(),
        // fullscreen: true,
        sample_count: 4,
        ..Default::default()
    }
}

async fn main_() {
    request_new_screen_size(600.0, 600.0);

    let (send, recv) = std::sync::mpsc::sync_channel(0);

    let mut points = Vec::new();

    while points.len() < 10 {
        let x = rand::gen_range(-128, 127i16);
        let y = rand::gen_range(-128, 127i16);
        points.push((x, y));
        points.sort();
        points.dedup();
    }

    // let points = [(4, 8), (8, 8), (8, 11), (6, 12)];
    // let points = [(900, 400), (910, 400), (900, 700), (910, 700), (0, 800)];

    let bounds = points
        .iter()
        .copied()
        .map(|(x, y)| (x, y, y, x))
        .reduce(|(l, t, r, b), (x, y, _, _)| (l.min(x), t.min(y), r.max(x), b.max(y)))
        .unwrap();

    let side = (bounds.2 - bounds.0).max(bounds.3 - bounds.1) as f32;

    let width = 3.0;
    let height = 3.0;

    let sites = &points
        .iter()
        .map(|(x, y)| {
            Point::new(
                width * (x - bounds.0) as f32 / side,
                height * (y - bounds.1) as f32 / side,
            )
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

    let camera = Camera2D {
        zoom: vec2(1.0 / width, -1.0 / height),
        target: vec2(width / 2.0, height / 2.0),
        ..Default::default()
    };

    let (mut benchline, mut events) = recv.recv().unwrap();

    set_camera(&camera);

    let bottom_left = camera.screen_to_world(vec2(0.0, 0.0));
    let top_right = camera.screen_to_world(vec2(screen_width(), screen_height()));
    let view = Rect {
        x: bottom_left.x,
        y: top_right.y,
        w: top_right.x - bottom_left.x,
        h: bottom_left.y - top_right.y,
    };

    let mut cells = None;

    loop {
        if is_key_pressed(KeyCode::Q) {
            return;
        }

        if is_key_pressed(KeyCode::Space) {
            if let Ok(b) = recv.try_recv() {
                (benchline, events) = b;
            }

            if _thread.as_ref().is_some_and(|x| x.is_finished()) {
                cells = Some(_thread.take().unwrap().join().unwrap());
                println!("cells: {:?}", cells);
            }
        }

        clear_background(LIGHTGRAY);

        for (i, point) in sites.iter().enumerate() {
            draw_circle(point.x, point.y, 0.08, RED);
            for other in (i + 1)..sites.len() {
                // draw bisector
                let b = Bisector::new(sites, i as SiteIdx, other as SiteIdx);
                draw_mediatriz(b, sites, view, GRAY);
            }
        }

        let mut hyperbola_colors = [GREEN, GREEN];

        for p in benchline.get_regions() {
            let p = sites[p as usize];
            draw_circle(p.x, p.y, 0.1, BLUE);
        }

        for b in benchline.get_bisectors() {
            let step = (view.right() - view.left()) / 100.0;

            draw_mediatriz(b, sites, view, BLUE);
            // draw_line(b.a.x, b.a.y, b.b.x, b.b.y, 0.02, BLUE);

            hyperbola_colors.swap(0, 1);
            let hcolor = hyperbola_colors[0];

            draw_hyperbola(view, step, b, sites, hcolor);
        }

        for v in events.iter() {
            match v {
                Event::Site(p) => {
                    let p = &sites[*p as usize];
                    draw_circle(p.x, p.y, 0.05, BLACK);
                }
                Event::Intersection(p, _) => {
                    draw_circle(p.x, p.y, 0.05, BLACK);
                    draw_line(p.x, view.top(), p.x, view.bottom(), 0.02, BLACK);
                }
            }
        }

        if let Some(cells) = &cells {
            draw_diagram(view, cells, sites).await;
        }

        next_frame().await
    }
}

fn draw_hyperbola(view: Rect, step: f32, b: Bisector, sites: &[Point], hcolor: Color) {
    let (q, r) = (sites[b.a as usize], sites[b.b as usize]);
    if q.y == r.y {
        let x = (q.x + r.x) / 2.0;
        draw_line(x, q.y, x, view.bottom(), 0.03, hcolor);
        return;
    }

    for i in 0..100 {
        let mut x1 = view.left() + i as f32 * step;
        let mut x2 = view.left() + (i + 1) as f32 * step;

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
        if (y1 > view.bottom() || y1 < view.top()) && (y2 > view.bottom() || y2 < view.top()) {
            continue;
        }
        draw_line(x1, y1, x2, y2, 0.03, hcolor);
    }
}

fn draw_mediatriz(b: Bisector, sites: &[Point], view: Rect, color: Color) {
    let (q, r) = (sites[b.a as usize], sites[b.b as usize]);
    if q.y == r.y {
        let x = (q.x + r.x) / 2.0;
        draw_line(x, view.top(), x, view.bottom(), 0.02, color);
        return;
    }

    let y0 = b.y_at(sites, view.left());
    let y1 = b.y_at(sites, view.right());
    draw_line(view.left(), y0, view.right(), y1, 0.02, color);
}

async fn draw_diagram(view: Rect, cells: &[Cell], sites: &[Point]) {
    let left = view.left();
    let right = view.right();
    for (c, cell) in cells.iter().enumerate() {
        for i in 0..cell.points.len() {
            let site = sites[c];
            let d = cell.points[(i + cell.points.len() - 1) % cell.points.len()];
            let a = cell.points[i];
            let b = cell.points[(i + 1) % cell.points.len()];

            if a.is_nan() && b.is_nan() {
                {
                    let n = cell.neighbors[i];
                    let bnc = Bisector::new(sites, c as SiteIdx, n);

                    let (p1, p2);
                    if site.y == sites[n as usize].y {
                        p1 = vec2((site.x + sites[n as usize].x) / 2.0, view.top());
                        p2 = vec2((site.x + sites[n as usize].x) / 2.0, view.bottom());
                    } else {
                        p1 = vec2(left, bnc.y_at(sites, left));
                        p2 = vec2(right, bnc.y_at(sites, right));
                    }
                    draw_line(p1.x, p1.y, p2.x, p2.y, 0.02, RED);
                }
                {
                    let n = cell.neighbors[(i + 1) % cell.points.len()];
                    let bnc = Bisector::new(sites, c as SiteIdx, n);

                    let (p1, p2);
                    if site.y == sites[n as usize].y {
                        p1 = vec2((site.x + sites[n as usize].x) / 2.0, view.top());
                        p2 = vec2((site.x + sites[n as usize].x) / 2.0, view.bottom());
                    } else {
                        p1 = vec2(left, bnc.y_at(sites, left));
                        p2 = vec2(right, bnc.y_at(sites, right));
                    }
                    draw_line(p1.x, p1.y, p2.x, p2.y, 0.02, RED);
                }
            } else if a.is_nan() {
                {
                    let n = cell.neighbors[i];
                    let bnc = Bisector::new(sites, c as SiteIdx, n);

                    let (p1, p2);
                    if site.y == sites[n as usize].y {
                        p1 = Point::new((site.x + sites[n as usize].x) / 2.0, view.top());
                        p2 = Point::new((site.x + sites[n as usize].x) / 2.0, view.bottom());
                    } else {
                        p1 = Point::new(left, bnc.y_at(sites, left));
                        p2 = Point::new(right, bnc.y_at(sites, right));
                    }

                    if (sites[n as usize] - site).perp_dot(p1 - site) > 0.0 {
                        draw_line(p1.x, p1.y, d.x, d.y, 0.02, RED);
                    } else {
                        draw_line(p2.x, p2.y, d.x, d.y, 0.02, RED);
                    }
                }
                {
                    let n = cell.neighbors[(i + 1) % cell.points.len()];
                    let bnc = Bisector::new(sites, c as SiteIdx, n);

                    let (p1, p2);
                    if site.y == sites[n as usize].y {
                        p1 = Point::new((site.x + sites[n as usize].x) / 2.0, view.top());
                        p2 = Point::new((site.x + sites[n as usize].x) / 2.0, view.bottom());
                    } else {
                        p1 = Point::new(left, bnc.y_at(sites, left));
                        p2 = Point::new(right, bnc.y_at(sites, right));
                    }

                    if (sites[n as usize] - site).perp_dot(p1 - site) < 0.0 {
                        draw_line(p1.x, p1.y, b.x, b.y, 0.02, RED);
                    } else {
                        draw_line(p2.x, p2.y, b.x, b.y, 0.02, RED);
                    }
                }
            } else if b.is_nan() {
            } else {
                draw_line(a.x, a.y, b.x, b.y, 0.02, RED);
                draw_circle(a.x, a.y, 0.05, RED);
            }
        }
    }
}
