use rand::{Rng, SeedableRng};
use voronoi_sweep::{fortune_algorithm, Point};

fn random_points() {
    let mut rng = rand::rngs::SmallRng::from_seed([76; 32]); // chosen by fair dice roll
    let sites = (0..512)
        .map(|_| Point::new(rng.gen(), rng.gen()))
        .collect::<Vec<_>>();

    for _ in 0..100 {
        fortune_algorithm(&sites, &mut |_, _| {});
    }
}

fn main() {
    let start = std::time::Instant::now();
    random_points();
    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);
}
