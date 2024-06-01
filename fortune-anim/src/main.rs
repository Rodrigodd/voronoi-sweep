use macroquad::prelude::*;

use voronoi_sweep::{fortune_algorithm, Beachline, Bisector, Cell, Event, Point, SiteIdx};

#[derive(PartialEq, Debug, Clone)]
enum AnimState {
    Running,
    TransitionPrelude { delay: f32, pos: Point },
    TransitionPostlude { delay: f32, pos: Point },
}

struct Style {
    prelude_delay: f32,
    postlude_delay: f32,
    sweepline_speed: f32,
    cell_colors: Vec<Color>,
    cell_border_thickness: f32,
    cell_border_color: Color,
    parabola_thickness: f32,
    parabola_color: Color,
    sweepline_thickness: f32,
    sweepline_color: Color,
    hyperbola_thickness: f32,
    hyperbola_color: Color,
    hyperbola_dot_thickness: f32,
    hyperbola_dot_color: Color,
    event_thickness: f32,
    event_color: Color,
    event_circle_radius: f32,
    event_circle_thickness: f32,
    event_circle_color: Color,
    site_thickness: f32,
    site_color: Color,
    background_color: Color,
}

#[inline_tweak::tweak_fn]
fn s() -> Style {
    Style {
        prelude_delay: 0.2,
        postlude_delay: 0.1,
        sweepline_speed: 0.5,
        cell_colors: [
            0x249200, //
            0x499D12, //
            0x5CA21B, //
            0x6EA824, //
            0x81AD2D, //
        ]
        .into_iter()
        .map(Color::from_hex)
        .collect(),
        cell_border_thickness: 0.02,
        cell_border_color: Color::from_hex(0x102010),
        parabola_thickness: 0.02,
        parabola_color: Color::from_hex(0x001500),
        sweepline_thickness: 0.02,
        sweepline_color: Color::from_hex(0x000000),
        hyperbola_thickness: 0.03,
        hyperbola_color: Color::from_hex(0x00a320),
        hyperbola_dot_thickness: 0.04,
        hyperbola_dot_color: Color::from_hex(0x00a320),
        event_thickness: 0.04,
        event_color: Color::from_hex(0x000000),
        event_circle_radius: 0.25,
        event_circle_thickness: 0.03,
        event_circle_color: Color::from_hex(0xd61928),
        site_thickness: 0.04,
        site_color: Color::from_hex(0xd61928),
        background_color: Color::from_hex(0xb9eadd),
    }
}

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

    while points.len() < 8 {
        let x = rand::gen_range(-128, 127i16);
        let y = rand::gen_range(-128, 80i16);
        points.push((x, y));
        points.sort();
        points.dedup();
    }

    // let points = [(0, 0), (1, 1), (2, 2), (3, 3)];

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
                height * ((y - bounds.1) as f32) / side,
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
    let mut paused = false;

    let mut anim_state = AnimState::Running;

    loop {
        let s = s();

        if is_key_pressed(KeyCode::Q) {
            return;
        }
        if is_key_pressed(KeyCode::R) {
            reset(&mut step, &steps, &mut sweepline, sites);
        }

        if mouse_wheel().1 != 0.0 {
            if mouse_wheel().1 > 0.0 {
                camera.zoom *= 1.1;
            } else {
                camera.zoom /= 1.1;
            }
        }

        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        if is_key_pressed(KeyCode::Left) {
            set_state(
                &steps,
                step.saturating_sub(2),
                &mut step,
                &mut sweepline,
                sites,
            );
            anim_state = AnimState::Running;
        }

        if is_key_pressed(KeyCode::Right) {
            match &mut anim_state {
                AnimState::Running => {
                    let this = steps[step].1.first();
                    if let Some(val) = this {
                        sweepline = val.pos(sites).y
                    }
                }
                AnimState::TransitionPrelude { delay, .. } => {
                    *delay = 0.0;
                }
                AnimState::TransitionPostlude { delay, .. } => {
                    *delay = 0.0;
                }
            }
        }

        match &mut anim_state {
            AnimState::Running => {
                if !paused {
                    sweepline += get_frame_time() * s.sweepline_speed;
                }
                if steps[step]
                    .1
                    .first()
                    .is_some_and(|e| e.pos(sites).y <= sweepline)
                {
                    sweepline = steps[step].1.first().unwrap().pos(sites).y;
                    anim_state = AnimState::TransitionPrelude {
                        delay: s.prelude_delay,
                        pos: steps[step].1.first().unwrap().pos(sites),
                    };
                }
            }
            AnimState::TransitionPrelude { delay, pos } => {
                if !paused {
                    *delay -= get_frame_time();
                }
                if *delay <= 0.0 {
                    set_state(&steps, step + 1, &mut step, &mut sweepline, sites);
                    anim_state = AnimState::TransitionPostlude {
                        delay: s.postlude_delay,
                        pos: *pos,
                    }
                }
            }
            AnimState::TransitionPostlude { delay, pos } => {
                if !paused {
                    *delay -= get_frame_time();
                }
                if *delay <= 0.0 {
                    sweepline = pos.y;
                    anim_state = AnimState::Running;
                }
            }
        }

        clear_background(s.background_color);

        let (beachline, events, cells) = steps.get(step).unwrap_or_else(|| steps.last().unwrap());
        draw_animation(
            &camera,
            view,
            cells,
            sites,
            beachline,
            sweepline,
            s,
            events,
            anim_state.clone(),
        );
        draw_ui(anim_state.clone());

        next_frame().await
    }
}

fn set_state(
    steps: &[(Beachline, Vec<Event>, Vec<Cell>)],
    index: usize,
    step: &mut usize,
    sweepline: &mut f32,
    sites: &[Point],
) {
    if index == 0 {
        *step = 0;
        *sweepline = 0.0;
    } else if index < steps.len() {
        *step = index;
        *sweepline = steps[index.saturating_sub(1)].1[0].pos(sites).y
    } else {
        *step = steps.len() - 1;
        *sweepline = steps[index].1[0].pos(sites).y
    }
}

fn reset(
    step: &mut usize,
    steps: &[(Beachline, Vec<Event>, Vec<Cell>)],
    sweepline: &mut f32,
    sites: &[Point],
) {
    set_state(steps, 0, step, sweepline, sites);
}

fn draw_ui(anim_state: AnimState) {
    set_default_camera();
    let fps = get_fps();
    draw_text(&format!("FPS: {:.2}", fps), 10.0, 10.0, 20.0, BLACK);
    draw_text(
        &format!("Transition time: {:.2?}", anim_state),
        10.0,
        30.0,
        20.0,
        BLACK,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_animation(
    camera: &Camera2D,
    view: Rect,
    cells: &[Cell],
    sites: &[Point],
    benchline: &Beachline,
    sweepline: f32,
    s: Style,
    events: &[Event],
    anim_state: AnimState,
) {
    set_camera(camera);

    draw_diagram_partial(view, cells, sites, benchline, sweepline);

    draw_line(
        view.left(),
        sweepline,
        view.right(),
        sweepline,
        s.sweepline_thickness,
        s.sweepline_color,
    );

    for point in sites.iter() {
        draw_circle(point.x, point.y, s.site_thickness, s.site_color);
    }

    for i in 0..benchline.len() {
        let p_idx = benchline.get_region(i);
        let left = benchline.get_left_boundary(i);
        let right = benchline.get_right_boundary(i);
        draw_parabola(view, sites, p_idx, sweepline, left, right);
    }

    for b in benchline.get_bisectors() {
        // draw_mediatriz(b, sites, view, BLUE);
        // draw_line(b.a.x, b.a.y, b.b.x, b.b.y, 0.02, BLUE);
        draw_hyperbola(view, b, sites, sweepline);
    }

    for v in events.iter() {
        match v {
            Event::Site(p) => {
                let p = &sites[*p as usize];
                draw_circle(p.x, p.y, s.event_thickness, s.event_color);
            }
            Event::Intersection(p, _) => {
                draw_circle(p.x, p.y, s.event_thickness, s.event_color);
            }
        }
    }

    if let AnimState::TransitionPrelude { pos, .. } | AnimState::TransitionPostlude { pos, .. } =
        anim_state
    {
        draw_poly_lines(
            pos.x,
            pos.y,
            64,
            s.event_circle_radius,
            0.0,
            s.event_circle_thickness,
            s.event_circle_color,
        );
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

    let s = s();

    if p.y == sweepline {
        // println!("p.y == sweepline {} {}", p.x, y_at(p.x));
        let y = left_boundary
            .or(right_boundary)
            .map_or(f32::NEG_INFINITY, |b| b.y_at(sites, p.x));
        draw_line(p.x, p.y, p.x, y, s.parabola_thickness, s.parabola_color);
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

        draw_line(x1, y1, x2, y2, s.parabola_thickness, s.parabola_color);
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

fn draw_hyperbola(view: Rect, b: Bisector, sites: &[Point], sweepline: f32) {
    let s = s();
    let (q, r) = (sites[b.a as usize], sites[b.b as usize]);
    if q.y == r.y {
        let x = (q.x + r.x) / 2.0;
        draw_line(
            x,
            q.y.max(sweepline),
            x,
            view.bottom(),
            s.hyperbola_thickness,
            s.hyperbola_color,
        );
        draw_circle(
            x,
            q.y.max(sweepline),
            s.hyperbola_dot_thickness,
            s.hyperbola_dot_color,
        );
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

        if y1 <= sweepline {
            let x = b.x_at_y_star(sites, sweepline);
            draw_line(
                x,
                sweepline,
                x2,
                y2,
                s.hyperbola_thickness,
                s.hyperbola_color,
            );
            draw_circle(
                x,
                sweepline,
                s.hyperbola_dot_thickness,
                s.hyperbola_dot_color,
            );
        } else if y2 <= sweepline {
            let x = b.x_at_y_star(sites, sweepline);
            draw_line(
                x,
                sweepline,
                x1,
                y1,
                s.hyperbola_thickness,
                s.hyperbola_color,
            );
            draw_circle(
                x,
                sweepline,
                s.hyperbola_dot_thickness,
                s.hyperbola_dot_color,
            );
        } else {
            draw_line(x1, y1, x2, y2, s.hyperbola_thickness, s.hyperbola_color);
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

        let s = s();
        let color = s.cell_colors[this_idx as usize % s.cell_colors.len()];

        // convert points, which delimits a convex polygon, to a Mesh
        draw_convex(&points, color);
        draw_lines(&points, s.cell_border_thickness, s.cell_border_color);
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
