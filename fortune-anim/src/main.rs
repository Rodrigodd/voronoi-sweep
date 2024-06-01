use macroquad::prelude::*;

use voronoi_sweep::{fortune_algorithm, Beachline, Bisector, Cell, Event, Point, SiteIdx};

const CELL_COLORS: [Color; 10] = {
    let mut x = [
        RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE, PINK, VIOLET, BROWN, DARKBLUE,
    ];
    let mut i = 0;
    while i < x.len() {
        x[i].a = 0.5;
        i += 1;
    }
    x
};

const FREEZE_DELAY: f32 = 0.1;

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

    let mut points = Vec::new();

    rand::srand(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    );

    while points.len() < 40 {
        let x = rand::gen_range(-128, 127i16);
        let y = rand::gen_range(-128, 127i16);
        points.push((x, y));
        points.sort();
        points.dedup();
    }

    // let points = [(0, 0), (4, 2), (8, 6), (0, 10)];

    let bounds = points
        .iter()
        .copied()
        .map(|(x, y)| (x, y, y, x))
        .reduce(|(l, t, r, b), (x, y, _, _)| (l.min(x), t.min(y), r.max(x), b.max(y)))
        .unwrap();

    let side = (bounds.2 - bounds.0).max(bounds.3 - bounds.1) as f32;
    let offset_x = (side - (bounds.2 - bounds.0) as f32) / 2.0;

    let width = 3.0;
    let height = 3.0;

    let sites = &points
        .iter()
        .map(|(x, y)| {
            Point::new(
                width * ((x - bounds.0) as f32 + offset_x) / side,
                height * (y - bounds.1) as f32 / side,
            )
        })
        .collect::<Vec<_>>();

    let (steps, diagram) = {
        let mut steps = Vec::new();
        let diagram = fortune_algorithm(sites, &mut |benchline, events, cells| {
            steps.push((benchline.clone(), events.to_vec(), cells.to_vec()));
        });

        (steps, diagram)
    };

    let mut camera = Camera2D {
        zoom: vec2(1.5 / width, -1.5 / height),
        target: vec2(width / 2.0, height / 2.0),
        ..Default::default()
    };

    let mut step = 0;
    let (mut benchline, mut events, mut cells) = steps[0].clone();

    let bottom_left = camera.screen_to_world(vec2(0.0, 0.0));
    let top_right = camera.screen_to_world(vec2(screen_width(), screen_height()));
    let view = Rect {
        x: bottom_left.x,
        y: top_right.y,
        w: top_right.x - bottom_left.x,
        h: bottom_left.y - top_right.y,
    };

    // let mut cells = None;

    let mut sweepline = 0.0;
    let mut freeze = 0.0;
    let mut paused = false;

    loop {
        if is_key_pressed(KeyCode::Q) {
            return;
        }
        if is_key_pressed(KeyCode::R) {
            step = 0;
            (benchline, events, cells) = steps[0].clone();
            sweepline = 0.0;
            freeze = 0.0;
        }

        if mouse_wheel().1 != 0.0 {
            if mouse_wheel().1 > 0.0 {
                camera.zoom *= 1.1;
            } else {
                camera.zoom /= 1.1;
            }
        }

        if !paused {
            if freeze > 0.0 {
                freeze -= get_frame_time();
            }
            if freeze <= 0.0 {
                sweepline += get_frame_time() * 0.5;
            }
        }

        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        if is_key_pressed(KeyCode::Left) {
            if let Some(b) = steps.get(step - 2) {
                step -= 2;
                sweepline = events[0].pos(sites).y;
                (benchline, events, cells) = b.clone();
            } else {
                cells = diagram.clone();
                events.clear();
                println!("cells: {:?}", cells);
            }
            freeze = FREEZE_DELAY;
        }

        if is_key_pressed(KeyCode::Right)
            || events.first().is_some_and(|e| e.pos(sites).y < sweepline)
        {
            freeze = FREEZE_DELAY;
            if let Some(b) = steps.get(step + 1) {
                step += 1;
                sweepline = events[0].pos(sites).y;
                (benchline, events, cells) = b.clone();
            } else {
                cells = diagram.clone();
                events.clear();
                println!("cells: {:?}", cells);
            }
        }

        set_default_camera();

        clear_background(LIGHTGRAY);

        let fps = get_fps();
        draw_text(&format!("FPS: {:.2}", fps), 10.0, 10.0, 20.0, BLACK);

        set_camera(&camera);

        draw_diagram_partial(view, &cells, sites, &benchline, sweepline);

        for point in sites.iter() {
            draw_circle(point.x, point.y, 0.05, RED);
        }

        for i in 0..benchline.len() {
            let p_idx = benchline.get_region(i);
            let left = benchline.get_left_boundary(i);
            let right = benchline.get_right_boundary(i);
            draw_parabola(view, sites, p_idx, sweepline, left, right);
        }

        let mut hyperbola_colors = [GREEN, GREEN];

        for b in benchline.get_bisectors() {
            // draw_mediatriz(b, sites, view, BLUE);
            // draw_line(b.a.x, b.a.y, b.b.x, b.b.y, 0.02, BLUE);

            hyperbola_colors.swap(0, 1);
            let hcolor = hyperbola_colors[0];

            draw_hyperbola(view, b, sites, hcolor, sweepline);
        }

        draw_line(view.left(), sweepline, view.right(), sweepline, 0.02, BLACK);

        for v in events.iter() {
            match v {
                Event::Site(p) => {
                    let p = &sites[*p as usize];
                    draw_circle(p.x, p.y, 0.03, BLACK);
                }
                Event::Intersection(p, _) => {
                    draw_circle(p.x, p.y, 0.03, BLACK);
                }
            }
        }

        next_frame().await
    }
}

fn draw_parabola(
    view: Rect,
    sites: &[Point],
    p_idx: SiteIdx,
    sweepline: f32,
    left_boundary: Option<Bisector>,
    right_boundary: Option<Bisector>,
) {
    let p = sites[p_idx as usize];
    let y_at = |x: f32| parabola_equation(p, sweepline, x);

    if p.y == sweepline {
        // println!("p.y == sweepline {} {}", p.x, y_at(p.x));
        let y = left_boundary
            .or(right_boundary)
            .map_or(f32::NEG_INFINITY, |b| b.y_at(sites, p.x));
        draw_line(p.x, p.y, p.x, y, 0.03, BLACK);
        return;
    }

    let left = left_boundary
        .map(|b| b.x_at_y_star(sites, sweepline))
        .unwrap_or(f32::NEG_INFINITY)
        .max(view.left());
    let right = right_boundary
        .map(|b| b.x_at_y_star(sites, sweepline))
        .unwrap_or(f32::INFINITY)
        .min(view.right());

    if left > right {
        return;
    }

    const STEPS: usize = 100;

    let step = (right - left) / STEPS as f32;
    for i in 0..STEPS {
        let x1 = left + i as f32 * step;
        let x2 = left + (i + 1) as f32 * step;

        let y1 = y_at(x1);
        let y2 = y_at(x2);
        if !y1.is_finite() || !y2.is_finite() {
            continue;
        }
        if (y1 > view.bottom() || y1 < view.top()) && (y2 > view.bottom() || y2 < view.top()) {
            continue;
        }

        draw_line(x1, y1, x2, y2, 0.03, BLACK);
    }
}

/// return the y value of the parabola equidistant to `p` and the horizontal line `sweepline` at
/// `x`.
fn parabola_equation(p: Point, sweepline: f32, x: f32) -> f32 {
    let d = p.y - sweepline;
    let a = 1.0 / (2.0 * d);
    let b = -2.0 * p.x / (2.0 * d);
    let c = (p.x * p.x + p.y * p.y - sweepline * sweepline) / (2.0 * d);
    a * x * x + b * x + c
}

/// return the values of x of the parabola equidistant to `p` and the horizontal line `sweepline` at
/// `y`.
fn parabola_equation_at_x(p: Point, sweepline: f32, y: f32) -> (f32, f32) {
    let d = p.y - sweepline;
    let a = 1.0 / (2.0 * d);
    let b = -2.0 * p.x / (2.0 * d);
    let c = (p.x * p.x + p.y * p.y - sweepline * sweepline) / (2.0 * d);
    let delta = b * b - 4.0 * a * (c - y);
    if delta < 0.0 {
        return (f32::NAN, f32::NAN);
    }
    let x1 = (-b + delta.sqrt()) / (2.0 * a);
    let x2 = (-b - delta.sqrt()) / (2.0 * a);
    if x1 < x2 {
        (x1, x2)
    } else {
        (x2, x1)
    }
}

fn draw_hyperbola(view: Rect, b: Bisector, sites: &[Point], hcolor: Color, sweepline: f32) {
    let (q, r) = (sites[b.a as usize], sites[b.b as usize]);
    if q.y == r.y {
        let x = (q.x + r.x) / 2.0;
        draw_line(x, q.y.max(sweepline), x, view.bottom(), 0.03, hcolor);
        return;
    }

    let step = (view.right() - view.left()) / 100.0;

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

        if y1 < sweepline && y2 < sweepline {
            continue;
        }

        if y1 < sweepline {
            let x = b.x_at_y_star(sites, sweepline);
            draw_line(x, sweepline, x2, y2, 0.03, hcolor);
            draw_circle(x, sweepline, 0.04, hcolor);
        } else if y2 < sweepline {
            let x = b.x_at_y_star(sites, sweepline);
            draw_line(x, sweepline, x1, y1, 0.03, hcolor);
            draw_circle(x, sweepline, 0.04, hcolor);
        } else {
            draw_line(x1, y1, x2, y2, 0.03, hcolor);
        }
    }
}

fn draw_diagram(view: Rect, cells: &[Cell], sites: &[Point]) {
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

fn draw_diagram_partial(
    view: Rect,
    cells: &[Cell],
    sites: &[Point],
    beachline: &Beachline,
    sweepline: f32,
) {
    for (c, cell) in cells.iter().enumerate() {
        let this_idx = c as SiteIdx;
        let mut points = Vec::new();

        // get parabola points
        for i in (0..beachline.len()).rev() {
            let reg = beachline.get_region(i);
            if reg != this_idx {
                continue;
            }
            let right = beachline
                .get_right_boundary(i)
                .map(|b| b.x_at_y_star(sites, sweepline));
            let left = beachline
                .get_left_boundary(i)
                .map(|b| b.x_at_y_star(sites, sweepline));

            let range = left.unwrap_or(f32::NEG_INFINITY)..right.unwrap_or(f32::INFINITY);

            let (r, l) = match (right, left) {
                (Some(r), Some(l)) => (r.min(view.right()), l.max(view.left())),
                (Some(r), None) => (r.min(view.right()), view.left().min(r)),
                (None, Some(l)) => (view.right().max(l), l.max(view.left())),
                (None, None) => (view.right(), view.left()),
            };

            let (x1, x2) = parabola_equation_at_x(sites[this_idx as usize], sweepline, view.top());

            if let Some(r) = right {
                points.push(Point::new(
                    r,
                    parabola_equation(sites[this_idx as usize], sweepline, r),
                ));
            } else if range.contains(&x2) {
                points.push(Point::new(x2, view.top()));
            }

            const STEPS: usize = 100;
            for i in 0..=STEPS {
                if l >= r {
                    break;
                }
                let linear_map = |t, a, b, x, y| {
                    let t = (t - a) / (b - a);
                    x + t * (y - x)
                };
                let x = linear_map(i as f32, 0.0, STEPS as f32, r, l);
                let y = parabola_equation(sites[this_idx as usize], sweepline, x);
                points.push(Point::new(x, y))
            }
            if let Some(l) = left {
                points.push(Point::new(
                    l,
                    parabola_equation(sites[this_idx as usize], sweepline, l),
                ));
            } else if range.contains(&x1) {
                points.push(Point::new(x1, view.top()));
            }
        }

        // get cell points
        for i in 0..cell.points.len() {
            let after = cell.points[i];
            if !after.is_nan() {
                points.push(after);
                // draw_circle(after.x, after.y, 0.05, RED);
            }

            // if the bisector is not bounded by two points, add the intersection of the bisector
            // with the bottom of the view.
            {
                let b_idx = cell.neighbors[i % cell.points.len()];
                let bisector = Bisector::new(sites, c as SiteIdx, b_idx);
                let x = bisector.x_at_y(sites, view.top());
                let intersection = Point::new(x, view.top());
                let cross = (sites[b_idx as usize] - sites[this_idx as usize])
                    .perp_dot(intersection - sites[this_idx as usize]);
                let parabola_y = parabola_equation(sites[this_idx as usize], sweepline, x);

                let prev = cell.points[(i + cell.points.len() - 1) % cell.points.len()];
                if x.is_finite()
                    && (cross >= 0.0 && after.is_nan() || cross <= 0.0 && prev.is_nan())
                    && parabola_y > view.top()
                {
                    // points.push(intersection);
                }
            }
        }

        if points.len() < 3 {
            continue;
        }

        let this = sites[this_idx as usize];
        points.sort_by(|a, b| voronoi_sweep::vec2_angle_cmp(*a - this, *b - this));

        let color = CELL_COLORS[this_idx as usize % CELL_COLORS.len()];

        // convert points, which delimits a convex polygon, to a Mesh
        draw_convex(&points, color);
        draw_lines(&points, 0.02, BLACK);
    }
}

/// Draw a solid convex polygon. The points must be sorted in counter-clockwise order.
fn draw_convex(points: &[Point], color: Color) {
    let triangles = (1..points.len() - 1)
        .flat_map(|i| {
            [
                0,
                (i) as u16 % points.len() as u16,
                (i + 1) as u16 % points.len() as u16,
            ]
        })
        .collect::<Vec<_>>();
    let mesh = Mesh {
        vertices: points
            .iter()
            .map(|p| macroquad::models::Vertex {
                position: vec3(p.x, p.y, 0.0),
                uv: vec2(0.0, 0.0),
                color,
            })
            .collect(),
        indices: triangles,
        texture: None,
    };
    draw_mesh(&mesh);
}

fn draw_lines(points: &[Point], thickness: f32, color: Color) {
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        draw_line(a.x, a.y, b.x, b.y, thickness, color);
    }
}
