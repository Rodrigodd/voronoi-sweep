use super::*;
use proptest::prelude::*;
use std::sync::atomic::{AtomicIsize, Ordering};

thread_local! {
    static ALLOC_COUNT: AtomicIsize = const { AtomicIsize::new(0) };
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct TestItem {
    value: i32,
}
impl TestItem {
    fn new(x: i32) -> Self {
        ALLOC_COUNT.with(|x| x.fetch_add(1, Ordering::Relaxed));
        debugln!("allocating {:?}", x);
        Self { value: x }
    }
}
impl Clone for TestItem {
    fn clone(&self) -> Self {
        Self::new(self.value)
    }
}
impl Drop for TestItem {
    fn drop(&mut self) {
        ALLOC_COUNT.with(|x| x.fetch_sub(1, Ordering::Relaxed));
        debugln!("dropping {:?}", self.value);
        // flush stdout
        // use std::io::Write;
        // let mut stdout = std::io::stdout().lock();
        // stdout.flush().unwrap();
        // writeln!(stdout, "dropping {:?}", self.0).unwrap();
        // stdout.flush().unwrap();
    }
}
impl std::fmt::Debug for TestItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

struct CheckAllocsOnDrop;
impl Drop for CheckAllocsOnDrop {
    fn drop(&mut self) {
        ALLOC_COUNT.with(|x| {
            assert_eq!(x.load(Ordering::Relaxed), 0);
        });
    }
}

#[test]
fn test_insert() {
    let mut tree = BTree::<i32, 3>::new();
    let cmp = i32::cmp;

    tree.insert(30, cmp);
    debugln!("{:?}", tree);
    tree.insert(10, cmp);
    debugln!("{:?}", tree);
    tree.insert(20, cmp);
    debugln!("{:?}", tree);
    tree.insert(60, cmp);
    debugln!("{:?}", tree);
    tree.insert(40, cmp);
    debugln!("{:?}", tree);
    tree.insert(50, cmp);
    debugln!("{:?}", tree);
    tree.insert(30, cmp);
    debugln!("{:?}", tree);

    debugln!("{:?}", tree.values());
}

#[test]
fn prop_fuzz_insertion() {
    let shuffle = (0..128u16)
        .prop_map(|n| (0..n as i32).collect::<Vec<i32>>())
        .prop_shuffle();

    let mut runner = proptest::test_runner::TestRunner::default();

    runner
        .run(&shuffle, |values| {
            debugln!("{:?}", values);
            check_sort(values);
            Ok(())
        })
        .unwrap();
}

#[test]
fn case1() {
    check_sort((0..10).collect());
}

#[test]
fn case2() {
    check_sort(vec![
        0, 1, 2, 3, 4, 5, 6, 9, 8, 7, 12, 11, 10, 15, 16, 14, 17, 13,
    ]);
}

#[test]
fn case3() {
    check_sort(vec![10, 4, 8, 6, 3, 1, 7, 9, 0, 5, 2]);
}

#[test]
fn case4() {
    check_sort(vec![0, 4, 8, 1, 9, 10, 3, 2, 7, 5, 6]);
}

fn check_sort(values: Vec<i32>) {
    let values = values
        .into_iter()
        .map(|x| TestItem::new(x))
        .collect::<Vec<_>>();

    let mut tree = BTree::<TestItem, 3>::new();
    let cmp = TestItem::cmp;

    debugln!("{:?}", tree);
    for value in &values {
        tree.insert(value.clone(), cmp);
        debugln!("{:?}", tree);
    }

    let mut values = values.clone();
    values.sort();
    let tree_values = tree.values();
    assert_eq!(values, tree_values);
}

#[test]
fn prop_fuzz_removal() {
    let shuffle = (0..64u16)
        .prop_map(|n| {
            let mut x = (0..n as i32)
                .map(|x| (x, true))
                .collect::<Vec<(i32, bool)>>();
            x.extend((0..n as i32).map(|x| (x, false)));
            x
        })
        .prop_shuffle();

    let mut runner = proptest::test_runner::TestRunner::default();
    runner
        .run(&shuffle, |values| {
            println!("{:?}", values);
            check_removal(values);
            Ok(())
        })
        .unwrap();
}

#[test]
#[ignore]
fn prop_fuzz_removal_parallel() {
    let (send, recv) = std::sync::mpsc::sync_channel(0);
    let threads = 4; // std::thread::available_parallelism().map_or(4, |x| x.get());
    println!("threads: {}", threads);
    for _ in 0..threads {
        let send = send.clone();
        std::thread::spawn(move || match std::panic::catch_unwind(prop_fuzz_removal) {
            Ok(_) => {}
            Err(_) => send.send(()).unwrap(),
        });
    }

    drop(send);
    let _ = recv.recv();
}

#[test]
fn removal_case1() {
    check_removal(vec![
        (0, false),
        (1, false),
        (2, false),
        (3, false),
        (0, true),
        (1, true),
        (2, true),
        (3, true),
    ]);
}

#[test]
fn removal_case2() {
    check_removal(vec![
        (1, false),
        (0, false),
        (2, false),
        (6, false),
        (1, true),
        (5, false),
        (3, false),
        (0, true),
        (5, true),
        (4, false),
        (6, true),
        (3, true),
        (4, true),
        (2, true),
    ])
}

#[test]
fn removal_case3() {
    check_removal(vec![
        (2, false),
        (1, false),
        (3, false),
        (4, false),
        (4, true),
        (2, true),
        (0, false),
    ])
}

#[test]
fn removal_case4() {
    check_removal(vec![
        (4, false),
        (1, false),
        (0, false),
        (6, false),
        (7, false),
        (5, false),
        (0, true),
        (3, false),
        (2, false),
        (3, true),
        (6, true),
        (1, true),
        (4, true),
        (2, true),
    ])
}

#[test]
fn removal_case5() {
    check_removal(vec![
        (6, false),
        (7, false),
        (3, false),
        (0, false),
        (3, true),
        (1, false),
        (4, false),
        (7, true),
        (2, false),
    ])
}

#[test]
fn removal_case6() {
    check_removal(vec![
        (2, false),
        (4, false),
        (1, false),
        (0, false),
        (3, false),
        (6, false),
        (7, false),
        (5, false),
        (10, false),
        (8, false),
        (9, false),
        (4, true),
        (3, true),
        (7, true),
        (5, true),
    ])
}

#[test]
fn removal_case7() {
    check_removal(vec![
        (7, false),
        (1, false),
        (8, false),
        (0, false),
        (6, false),
        (2, false),
        (5, false),
        (4, false),
        (10, false),
        (9, false),
        (8, true),
        (3, false),
        (4, true),
        (7, true),
        (6, true),
        (0, true),
    ])
}

#[test]
fn removal_case8() {
    check_removal(vec![
        (2, true),
        (0, true),
        (1, false),
        (8, false),
        (3, false),
        (8, true),
        (9, false),
        (2, false),
        (3, true),
        (7, false),
        (9, true),
        (5, true),
        (1, true),
        (6, false),
        (0, false),
        (5, false),
        (4, true),
        (6, true),
        (4, false),
        (7, true),
    ])
}

#[test]
fn removal_case9() {
    check_removal(vec![
        (20, false),
        (0, false),
        (0, true),
        (15, false),
        (13, false),
        (13, true),
        (1, true),
        (2, false),
        (18, true),
        (8, false),
        (6, false),
        (15, true),
        (22, true),
        (17, true),
        (21, false),
        (4, false),
        (9, false),
        (11, false),
        (3, false),
        (6, true),
        (4, true),
        (21, true),
        (9, true),
        (18, false),
        (12, false),
        (7, false),
        (1, false),
        (2, true),
        (14, true),
        (10, true),
        (8, true),
        (14, false),
        (17, false),
        (10, false),
        (5, false),
        (16, false),
        (22, false),
        (11, true),
        (19, false),
        (19, true),
        (7, true),
        (5, true),
    ])
}

#[test]
fn removal_case10() {
    check_removal(vec![
        (6, false),
        (5, false),
        (4, false),
        (5, true),
        (2, false),
        (9, false),
        (8, false),
        (3, false),
        (15, false),
        (0, false),
        (18, false),
        (13, false),
        (17, false),
        (10, false),
        (14, false),
        (12, false),
        (6, true),
        (7, false),
        (19, false),
        (7, true),
        (17, true),
        (21, false),
        (16, false),
        (4, true),
        (20, false),
        (8, true),
        (3, true),
        (1, false),
        (19, true),
        (11, false),
        (2, true),
        (15, true),
        (13, true),
    ])
}

fn check_removal(values: Vec<(i32, bool)>) {
    let _ondrop = CheckAllocsOnDrop;
    let values = values
        .into_iter()
        .map(|(x, b)| (TestItem::new(x), b))
        .collect::<Vec<_>>();

    let mut tree = BTree::<TestItem, 3>::new();
    let cmp = TestItem::cmp;

    let mut final_values = Vec::new();

    debugln!("{:?}", tree);
    for (value, remove) in &values {
        if *remove {
            let x = tree.remove(value, cmp);
            debugln!("remove {:?} {:?}", value, x);
            if let Some(i) = final_values.iter().position(|x| x == value) {
                final_values.remove(i);
            }
        } else {
            debugln!("insert {:?}", value);
            tree.insert(value.clone(), cmp);
            final_values.push(value.clone());
        }
        debugln!("{:?}", tree);
    }
    let mut values = final_values.into_iter().collect::<Vec<_>>();
    values.sort_by(cmp);
    let tree_values = tree.values();

    assert_eq!(values, tree_values);
}

#[test]
fn prop_fuzz_removal4() {
    let shuffle = (0..64u16)
        .prop_map(|n| {
            let mut x = (0..n as i32)
                .map(|x| (x, true))
                .collect::<Vec<(i32, bool)>>();
            x.extend((0..n as i32).map(|x| (x, false)));
            x
        })
        .prop_shuffle();

    let mut runner = proptest::test_runner::TestRunner::default();
    runner
        .run(&shuffle, |values| {
            println!("{:?}", values);
            check_removal4(values);
            Ok(())
        })
        .unwrap();
}

#[test]
fn removal4_case1() {
    check_removal4(vec![
        (0, true),
        (1, true),
        (3, true),
        (2, true),
        (5, true),
        (4, true),
        (5, false),
        (0, false),
        (4, false),
        (3, false),
        (2, false),
        (1, false),
    ])
}

#[test]
fn removal4_case2() {
    check_removal4(vec![
        (41, false),
        (35, false),
        (41, true),
        (16, true),
        (3, true),
        (39, false),
        (17, true),
        (19, false),
        (2, true),
        (21, false),
        (7, true),
        (3, false),
        (29, false),
        (16, false),
        (46, false),
        (39, true),
        (25, false),
        (48, false),
        (14, true),
        (49, false),
        (32, true),
        (42, true),
        (15, false),
        (13, false),
        (28, true),
        (7, false),
        (31, true),
        (0, true),
        (47, true),
        (26, true),
        (29, true),
        (37, true),
        (6, false),
        (47, false),
        (35, true),
        (10, false),
        (33, false),
        (36, false),
        (24, true),
        (20, false),
        (31, false),
        (0, false),
        (46, true),
        (10, true),
        (1, false),
        (44, false),
        (13, true),
        (23, true),
        (17, false),
        (14, false),
        (45, false),
        (22, true),
        (27, true),
        (15, true),
        (25, true),
        (45, true),
        (26, false),
        (9, true),
        (30, true),
        (44, true),
        (27, false),
        (9, false),
        (18, false),
        (5, true),
        (30, false),
        (4, true),
        (8, true),
        (40, true),
        (32, false),
        (36, true),
        (37, false),
        (11, true),
        (23, false),
        (43, false),
        (6, true),
        (22, false),
        (49, true),
        (33, true),
        (1, true),
        (42, false),
        (48, true),
        (12, true),
        (2, false),
        (40, false),
        (21, true),
        (8, false),
        (5, false),
        (38, true),
        (38, false),
        (11, false),
        (4, false),
        (43, true),
        (19, true),
        (24, false),
        (12, false),
        (28, false),
        (18, true),
        (34, true),
        (20, true),
        (34, false),
    ])
}

fn check_removal4(values: Vec<(i32, bool)>) {
    let _ondrop = CheckAllocsOnDrop;
    let values = values
        .into_iter()
        .map(|(x, b)| (TestItem::new(x), b))
        .collect::<Vec<_>>();

    let mut tree = BTree::<TestItem, 4>::new();
    let cmp = TestItem::cmp;

    let mut final_values = Vec::new();

    debugln!("{:?}", tree);
    for (value, remove) in &values {
        if *remove {
            let x = tree.remove(value, cmp);
            debugln!("remove {:?} {:?}", value, x);
            if let Some(i) = final_values.iter().position(|x| x == value) {
                final_values.remove(i);
            }
        } else {
            debugln!("insert {:?}", value);
            tree.insert(value.clone(), cmp);
            final_values.push(value.clone());
        }
        debugln!("{:?}", tree);
    }
    let mut values = final_values.into_iter().collect::<Vec<_>>();
    values.sort_by(cmp);
    let tree_values = tree.values();

    assert_eq!(values, tree_values);
}
