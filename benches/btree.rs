use std::collections::{BTreeMap, BTreeSet};
use voronoi_sweep::btree::BTree;

type Item = [u64; 2];

fn my_tree<const N: usize>(points: &[Item]) {
    let mut tree: BTree<Item, N> = BTree::new();

    for p in points {
        tree.insert(*p, Item::cmp);
    }

    for p in points {
        tree.remove(p, Item::cmp);
    }

    let tree = std::hint::black_box(tree);
    std::mem::forget(tree);
}

fn rust_tree(points: &[Item]) {
    let mut tree = BTreeSet::new();

    for p in points {
        tree.insert(*p);
    }

    for p in points {
        tree.remove(p);
    }

    let tree = std::hint::black_box(tree);
    std::mem::forget(tree);
}

macro_rules! bench {
    ($times:expr, $name:literal, $n:expr) => {
        let start = std::time::Instant::now();
        $n;
        let elapsed = start.elapsed();
        $times.entry($name).or_insert(vec![]).push(elapsed);
    };
}

fn main() {
    use rand::{Rng, SeedableRng};
    const N: u64 = 500_000;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);
    let points: Vec<Item> = (0..N).map(|_| rng.gen()).collect();

    let mut times = BTreeMap::new();

    for _ in 0..20 {
        bench!(times, "my_tree_06", my_tree::<6>(&points));
        bench!(times, "my_tree_07", my_tree::<7>(&points));
        bench!(times, "my_tree_08", my_tree::<8>(&points));
        bench!(times, "my_tree_11", my_tree::<11>(&points));
        bench!(times, "my_tree_12", my_tree::<12>(&points));
        bench!(times, "my_tree_15", my_tree::<15>(&points));
        bench!(times, "my_tree_16", my_tree::<16>(&points));
        bench!(times, "my_tree_23", my_tree::<23>(&points));
        bench!(times, "my_tree_24", my_tree::<24>(&points));
        bench!(times, "my_tree_31", my_tree::<31>(&points));
        bench!(times, "my_tree_32", my_tree::<32>(&points));
        bench!(times, "rust_tree", rust_tree(&points));
    }

    for (name, times) in times {
        let total = times.iter().sum::<std::time::Duration>();
        let min = times.iter().min().unwrap();
        let avg = total / times.len() as u32;
        let std_dev = (times
            .iter()
            .map(|t| (t.as_secs_f64() - avg.as_secs_f64()).powi(2))
            .sum::<f64>()
            / times.len() as f64)
            .sqrt();
        println!(
            "{:10}: min {:7.1?}, avg {:7.1?} ± {:7.2?}ms",
            name,
            min,
            avg,
            std_dev * 1e3
        );
    }
}
