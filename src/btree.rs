use super::debugln;
use std::{fmt::Debug, mem::MaybeUninit, ptr::NonNull};

const ORDER: usize = 3;
const CAPACITY: usize = ORDER - 1;

macro_rules! test_assert {
    ($cond:expr) => {
        #[cfg(debug_assertions)]
        {
            assert!($cond);
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            assert!($cond, $($arg)*);
        }
    };
}

trait OptionTestExt {
    fn test_assert_none(&self);
}
impl<T> OptionTestExt for Option<T> {
    fn test_assert_none(&self) {
        test_assert!(self.is_none());
    }
}

pub struct BTree<T> {
    root: InternalNode<T>,
    /// The depth of the tree. A tree where the root is a leaf has depth 0.
    depth: usize,
}
impl<T: Ord + Debug> BTree<T> {
    pub fn new() -> Self {
        BTree {
            root: InternalNode {
                inner: LeafNode::new(),
                children: [None; ORDER],
            },
            depth: 0,
        }
    }

    pub fn insert(&mut self, value: T) {
        debugln!("adding {:?}", value);
        if self.depth == 0 {
            debugln!("adding to leaf root");
            test_assert!(!self.root.inner.internal);
            let Some((median, right)) = self.root.inner.insert(value) else {
                return;
            };

            debugln!(
                "leaf roof split! {:?} {:?} {:?}",
                self.root.inner,
                median,
                right
            );

            let parent = InternalNode::new();
            let left = std::mem::replace(&mut self.root, parent).inner;

            self.root.inner.values[0].write(median);
            self.root.children[0] = NonNull::new(Box::into_raw(Box::new(left)).cast::<Node<T>>());
            self.root.children[1] = NonNull::new(Box::into_raw(Box::new(right)).cast::<Node<T>>());
            self.root.inner.len = 1;
            self.depth += 1;
        } else {
            debugln!("adding to internal root");
            // find child that cotains the value
            let Some((median, right)) = self.root.find_and_insert(value, self.depth) else {
                return;
            };

            debugln!(
                "internal roof split! {:?} {:?} {:?}",
                self.root,
                median,
                right
            );

            let parent = InternalNode::new();
            let left = std::mem::replace(&mut self.root, parent);

            self.root.inner.values[0].write(median);
            self.root.children[0] = NonNull::new(Box::into_raw(Box::new(left)).cast::<Node<T>>());
            self.root.children[1] = NonNull::new(Box::into_raw(Box::new(right)).cast::<Node<T>>());
            self.root.inner.len = 1;

            self.depth += 1;

            test_assert!(self.root.children[..self.root.inner.len]
                .iter()
                .all(|x| x.is_some()));
            test_assert!(self
                .root
                .children
                .get(self.root.inner.len + 1)
                .and_then(|c| c.as_ref())
                .is_none());
        }
    }
}
impl<T: Clone> BTree<T> {
    fn values(&self) -> Vec<T> {
        if self.depth == 0 {
            self.root.inner.values().cloned().collect::<Vec<T>>()
        } else {
            self.root.values(self.depth)
        }
    }
}
impl<T: Clone> InternalNode<T> {
    fn values(&self, depth: usize) -> Vec<T> {
        test_assert!(depth > 0);
        let mut values: Vec<T> = Vec::new();
        for (child, value) in self.children[..self.inner.len + 1]
            .iter()
            .zip(self.inner.values().map(Some).chain(std::iter::once(None)))
        {
            let child = child.expect("child is None");
            if depth > 1 {
                let child = unsafe { child.cast::<InternalNode<T>>().as_ref() };
                test_assert!(child.inner.internal, "child is internal");
                values.extend(child.values(depth - 1));
            } else {
                let child = unsafe { child.cast::<LeafNode<T>>().as_ref() };
                test_assert!(!child.internal, "child is leaf");
                values.extend(child.values().cloned());
            }

            if let Some(value) = value {
                values.push(value.clone());
            }
        }

        values
    }
}

/// A node is either a Leaf or an InternalNode. But a pointer to a InternalNode<T> is a valid
/// pointer to a LeafNode<T>.
#[repr(transparent)]
#[derive(Debug)]
struct Node<T>(LeafNode<T>);

struct LeafNode<T: Sized> {
    values: [MaybeUninit<T>; CAPACITY],
    len: usize,
    #[cfg(debug_assertions)]
    internal: bool,
}
impl<T: Ord + Debug> LeafNode<T> {
    fn new() -> Self {
        LeafNode {
            values: [(); CAPACITY].map(|_| MaybeUninit::uninit()),
            len: 0,
            #[cfg(debug_assertions)]
            internal: false,
        }
    }

    /// Inserts a key into the node. If the node is full, the node is splited, in that case self
    /// becomes the left node and the median and the right node are returned.
    fn insert(&mut self, value: T) -> Option<(T, Self)> {
        test_assert!(!self.internal);

        if self.len == CAPACITY {
            let (median, right) = self.split_at_median(value);

            debugln!("leaf splited {:?} {:?} {:?}", self, median, right);

            test_assert!(!self.internal);
            test_assert!(!right.internal);

            return Some((median, right));
        }
        let mut i = self.len;
        // SAFETY: All elements with index < self.len are initialized. This invariant is valid at
        // initialization, with self.len = 0, and is maintained by this method.
        unsafe {
            while i > 0 && *self.values[i - 1].assume_init_ref() > value {
                std::ptr::copy_nonoverlapping(
                    self.values[i - 1].as_ptr(),
                    self.values[i].as_mut_ptr(),
                    1,
                );
                i -= 1;
            }
        }
        self.values[i].write(value);
        self.len += 1;

        None
    }

    fn split_at_median(&mut self, value: T) -> (T, LeafNode<T>) {
        test_assert!(self.len == CAPACITY);

        let mut right = LeafNode::<T>::new();
        // let median;

        // check if we need to insert the value in the self or right node
        let middle = CAPACITY / 2;
        // SAFETY: self is fully initialized
        if unsafe { value < *self.values[middle].assume_init_ref() } {
            // we need to insert the value in the self node

            // move the right half of the keys to the right node
            //
            // SAFETY: all elements in self.values are initialized. The elements in
            // middle..CAPACITY are being moved to right.values[0..CAPACITY - middle]. The length
            // of both nodes are updated, according to the number of elements moved.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.values.as_ptr().add(middle),
                    right.values.as_mut_ptr(),
                    CAPACITY - middle,
                );
                self.len = middle;
                right.len = CAPACITY - middle;
            }

            // insert the value in the self node
            self.insert(value).test_assert_none();

            // SAFETY: after the insertion above, middle is initialized. We pop it out of
            // self, updating the length of self accordingly.
            unsafe {
                self.len = middle;
                (self.values[middle].assume_init_read(), right)
            }
        } else {
            // we need to insert the value in the right node

            // the median is the middle element
            // SAFETY: self is fully initialized, and middle will not be read anymore.
            let median = unsafe { self.values[middle].assume_init_read() };
            self.len = middle;

            // copy the right half of the keys to the right node
            // SAFETY: self is fully initialized, and all indexes < right.len become initialized.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.values.as_ptr().add(middle + 1),
                    right.values.as_mut_ptr(),
                    CAPACITY - (middle + 1),
                );
                right.len = CAPACITY - middle - 1;
            }

            // insert the value in the right node
            right.insert(value).test_assert_none();

            (median, right)
        }
    }
}
impl<T> LeafNode<T> {
    fn values(&self) -> impl Iterator<Item = &T> {
        // SAFETY: all indices smaller than self.len are initialized
        self.values[..self.len]
            .iter()
            .map(|v| unsafe { v.assume_init_ref() })
    }
}

#[repr(C)]
struct InternalNode<T> {
    inner: LeafNode<T>,
    children: [Option<NonNull<Node<T>>>; ORDER],
}
impl<T: Ord + Debug> InternalNode<T> {
    fn new() -> Self {
        InternalNode {
            inner: LeafNode {
                #[cfg(debug_assertions)]
                internal: true,
                ..LeafNode::new()
            },
            children: [None; ORDER],
        }
    }

    fn find_and_insert(&mut self, value: T, depth: usize) -> Option<(T, Self)> {
        // find which child to insert the value
        let child_idx = 'find_child: {
            for (i, item) in self.inner.values().enumerate() {
                if value < *item {
                    break 'find_child i;
                }
            }
            self.inner.len
        };

        // insert the value in the child
        let mut child = self.children[child_idx].expect("child is None");
        let (median, right) = if depth == 1 {
            // our child is a leaf
            let mut child = unsafe { child.cast::<LeafNode<T>>().as_mut() };
            test_assert!(!child.internal, "child is leaf");
            let (median, right) = child.insert(value)?;
            let right = NonNull::new(Box::into_raw(Box::new(right)).cast::<Node<T>>()).unwrap();
            (median, right)
        } else {
            // our child is an internal node
            let child: &mut InternalNode<T> = unsafe { child.cast::<InternalNode<T>>().as_mut() };
            test_assert!(child.inner.internal, "child is internal");
            let (median, right) = child.find_and_insert(value, depth - 1)?;
            let right = NonNull::new(Box::into_raw(Box::new(right)).cast::<Node<T>>()).unwrap();
            (median, right)
        };

        debugln!(
            "child splitted [{:?}] {:?} {:?}",
            child_idx,
            median,
            unsafe { right.as_ref() }
        );

        // self.insert(median, right)

        let Some((median, right)) = self.insert(median, right) else {
            return None;
        };

        use std::fmt;
        pub struct Fmt<F>(pub F)
        where
            F: Fn(&mut fmt::Formatter) -> fmt::Result;

        impl<F> fmt::Debug for Fmt<F>
        where
            F: Fn(&mut fmt::Formatter) -> fmt::Result,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                (self.0)(f)
            }
        }

        // debugln!(
        //     "splitted~ {:?} {:?} {:?}",
        //     Fmt(|f| self.format(f, depth - 1)),
        //     self,
        //     Fmt(|f| right.format(f, depth))
        // );

        Some((median, right))
    }

    /// Inserts a key into the node. If the node is full, the node is splited, in that case self
    /// becomes the left node and the median and the right node are returned.
    fn insert(&mut self, value: T, child: NonNull<Node<T>>) -> Option<(T, Self)> {
        debugln!("inserting {:?} {:?}", value, unsafe { child.as_ref() });

        if self.inner.len == CAPACITY {
            let (median, rigth) = self.split_at_median(value, child);
            debugln!("internal splited {:?} {:?} {:?}", self, median, rigth);

            return Some((median, rigth));
        }
        let mut i = self.inner.len;
        // SAFETY: All elements with index < self.len are initialized. This invariant is valid at
        // initialization, with self.len = 0, and is maintained by this method.
        unsafe {
            while i > 0 && *self.inner.values[i - 1].assume_init_ref() > value {
                std::ptr::copy_nonoverlapping(
                    self.inner.values[i - 1].as_ptr(),
                    self.inner.values[i].as_mut_ptr(),
                    1,
                );
                i -= 1;
            }
        }

        // move the child pointers
        self.children.copy_within(i + 1..self.inner.len + 1, i + 2);

        self.inner.values[i].write(value);
        self.children[i + 1] = Some(child);
        self.inner.len += 1;

        test_assert!(
            self.inner.len >= CAPACITY / 2,
            "invariant broken {:?} < {:?}",
            self.inner.len,
            CAPACITY / 2
        );
        test_assert!(self.children[..self.inner.len + 1]
            .iter()
            .all(|x| x.is_some()));

        None
    }

    fn split_at_median(&mut self, value: T, child: NonNull<Node<T>>) -> (T, InternalNode<T>) {
        test_assert!(self.inner.len == CAPACITY);

        let mut right = InternalNode::<T>::new();
        // let median;

        // check if we need to insert the value in the self or right node
        let middle = CAPACITY / 2;
        // SAFETY: self is fully initialized
        let (median, right) = if unsafe { value < *self.inner.values[middle].assume_init_ref() } {
            // we need to insert the value in the self node

            // SAFETY: all elements in self.values are initialized. The elements in
            // self.values[middle..CAPACITY] are being moved to right.values[0..CAPACITY - middle].
            // The elements in self.children[middle+1..CAPACITY+1] are being moved to
            // right.children[1..CAPACITY - middle + 1]. The length of both nodes are updated,
            // according to the number of elements moved.
            // After this, right.children[0] is still null.
            unsafe {
                // copy the right half of the keys to the right node
                std::ptr::copy_nonoverlapping(
                    self.inner.values.as_ptr().add(middle),
                    right.inner.values.as_mut_ptr(),
                    CAPACITY - middle,
                );
                right.inner.len = CAPACITY - middle;
                self.inner.len = middle;

                // copy children pointers
                right.children[1..right.inner.len + 1]
                    .copy_from_slice(&self.children[middle + 1..]);
            }

            // insert the value in the self node
            self.insert(value, child).test_assert_none();

            // SAFETY: pop the median out of self, updating the length of self accordingly, and
            // moving the popped child pointer to right.children[0].
            let median = unsafe {
                self.inner.len = middle;
                let median = self.inner.values[middle].assume_init_read();
                right.children[0] = self.children[middle + 1];
                median
            };

            (median, right)
        } else {
            // we need to insert the value in the right node

            // the median is the middle element
            // SAFETY: self is fully initialized, and middle will not be read anymore.
            let median = unsafe { self.inner.values[middle].assume_init_read() };
            self.inner.len = middle;

            // copy the right half of the keys to the right node
            // SAFETY: self is fully initialized, and all indexes < right.len become initialized.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.inner.values.as_ptr().add(middle + 1),
                    right.inner.values.as_mut_ptr(),
                    CAPACITY - (middle + 1),
                );
                right.inner.len = CAPACITY - middle - 1;
            }

            // copy children pointers
            right.children[..right.inner.len + 1].copy_from_slice(&self.children[middle + 1..]);

            // insert the value in the right node
            let mut i = right.inner.len;
            // SAFETY: All elements with index < self.len are initialized. This invariant is valid at
            // initialization, with self.len = 0, and is maintained by this method.
            unsafe {
                while i > 0 && *right.inner.values[i - 1].assume_init_ref() > value {
                    std::ptr::copy_nonoverlapping(
                        right.inner.values[i - 1].as_ptr(),
                        right.inner.values[i].as_mut_ptr(),
                        1,
                    );
                    right.children[i + 1] = right.children[i];
                    i -= 1;
                }
            }
            right.inner.values[i].write(value);
            right.children[i + 1] = Some(child);
            right.inner.len += 1;

            (median, right)
        };

        test_assert!(self.children[..self.inner.len + 1]
            .iter()
            .all(|x| x.is_some()));
        test_assert!(right.children[..right.inner.len + 1]
            .iter()
            .all(|x| x.is_some()));

        (median, right)
    }
}
impl<T: Debug> Debug for BTree<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BTree(")?;
        write!(f, "depth: {}, ", self.depth)?;
        if self.depth == 0 {
            self.root.inner.format(f)?;
        } else {
            self.root.format(f, self.depth)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl<T: Debug> Debug for InternalNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "~InternalNode(")?;
        self.inner.format(f)?;
        write!(f, ")")?;
        Ok(())
    }
}

impl<T: Debug> Debug for LeafNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LeafNode[")?;
        for (i, value) in self.values().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", value)?;
        }
        write!(f, "]")
    }
}

impl<T: Debug> InternalNode<T> {
    fn format(&self, f: &mut std::fmt::Formatter<'_>, depth: usize) -> std::fmt::Result {
        test_assert!(depth > 0);
        write!(f, "[").unwrap();
        for (child, value) in self.children[..self.inner.len + 1]
            .iter()
            .zip(self.inner.values().map(Some).chain(std::iter::once(None)))
        {
            let child = child.expect("child is None");
            if depth > 1 {
                let child = unsafe { child.cast::<InternalNode<T>>().as_ref() };
                test_assert!(child.inner.internal, "child is internal");
                child.format(f, depth - 1)?;
            } else {
                let child = unsafe { child.cast::<LeafNode<T>>().as_ref() };
                test_assert!(!child.internal, "child is leaf");
                child.format(f)?;
            }

            if let Some(value) = value {
                write!(f, " {:?} ", value)?;
            }
        }
        write!(f, "]").unwrap();
        Ok(())
    }
}

impl<T: Debug> LeafNode<T> {
    fn format(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, value) in self.values().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", value)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    #[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
    struct TestItem(i32);
    impl Drop for TestItem {
        fn drop(&mut self) {
            debugln!("dropping {:?}", self.0);
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
            self.0.fmt(f)
        }
    }

    #[test]
    fn test_insert() {
        let mut tree = BTree::<i32>::new();

        tree.insert(30);
        debugln!("{:?}", tree);
        tree.insert(10);
        debugln!("{:?}", tree);
        tree.insert(20);
        debugln!("{:?}", tree);
        tree.insert(60);
        debugln!("{:?}", tree);
        tree.insert(40);
        debugln!("{:?}", tree);
        tree.insert(50);
        debugln!("{:?}", tree);
        tree.insert(30);
        debugln!("{:?}", tree);

        debugln!("{:?}", tree.values());
    }

    #[test]
    fn prop_fuzz() {
        let shuffle = (0..128u16)
            .prop_map(|n| (0..n as i32).collect::<Vec<i32>>())
            .prop_shuffle();

        let mut runner = proptest::test_runner::TestRunner::default();

        runner
            .run(&shuffle, |values| {
                println!("{:?}", values);
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
        let values = values.into_iter().map(|x| TestItem(x)).collect::<Vec<_>>();

        let mut tree = BTree::<TestItem>::new();
        debugln!("{:?}", tree);
        for value in &values {
            tree.insert(value.clone());
            debugln!("{:?}", tree);
        }

        let mut values = values.clone();
        values.sort();
        let tree_values = tree.values();
        assert_eq!(values, tree_values);
    }
}
