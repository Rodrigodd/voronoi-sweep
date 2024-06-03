use std::time::Duration;

use rand::{Rng, SeedableRng};
use voronoi_sweep::{fortune_algorithm, Point};

fn random_points(number: usize) {
    let mut rng = rand::rngs::SmallRng::from_seed([76; 32]); // chosen by fair dice roll
    let sites = (0..number)
        .map(|_| Point::new(rng.gen(), rng.gen()))
        .collect::<Vec<_>>();

    for _ in 0..10 {
        fortune_algorithm(&sites, &mut |_, _, _| {});
    }
}

fn main() {
    let mut n = 1;
    println!(" i |     N |    elapsed | increase ");
    let mut times: Vec<Duration> = Vec::new();
    for i in 0..=14 {
        let start = std::time::Instant::now();
        random_points(n);
        // worst_case(n);
        let elapsed = start.elapsed();
        let increase = times
            .last()
            .map(|t| elapsed.as_secs_f64() / t.as_secs_f64())
            .unwrap_or(f64::NAN);
        println!("{:>2} | {:>5} | {:>10.3?} | {:.2}", i, n, elapsed, increase);

        times.push(elapsed);

        n *= 2;
    }
}
