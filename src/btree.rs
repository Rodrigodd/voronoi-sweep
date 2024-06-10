use super::debugln;
use std::cmp::Ordering;
use std::{fmt::Debug, mem::MaybeUninit, ptr::NonNull};

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

mod debug;

trait OptionTestExt {
    fn test_assert_none(&self);
}
impl<T> OptionTestExt for Option<T> {
    fn test_assert_none(&self) {
        test_assert!(self.is_none());
    }
}

/// A BTree, holding elements of type T, with a maximum of N elements per node and N+1 children per
/// node, with a custom comparator F.
pub struct BTree<T, const N: usize> {
    /// The root of the tree, is either a LeafNode or an InternalNode, if depth is 0 or not.
    root: InternalNode<T, N>,
    /// The depth of the tree. A tree where the root is a leaf has depth 0.
    depth: usize,
}
impl<T: Debug, const N: usize> BTree<T, N> {
    pub fn new() -> Self {
        BTree {
            root: InternalNode {
                inner: LeafNode::new(),
                child0: None,
                childs: [None; N],
            },
            depth: 0,
        }
    }

    pub fn insert<F: Fn(&T, &T) -> Ordering>(&mut self, value: T, cmp: F) {
        debugln!("adding {:?}", value);
        if self.depth == 0 {
            debugln!("adding to leaf root");
            test_assert!(!self.root.inner.internal);
            let Some((median, right)) = self.root.inner.insert(value, &cmp) else {
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
            self.root.children_mut()[0] =
                NonNull::new(Box::into_raw(Box::new(left)).cast::<Node<T, N>>());
            self.root.children_mut()[1] =
                NonNull::new(Box::into_raw(Box::new(right)).cast::<Node<T, N>>());
            self.root.inner.len = 1;
            self.depth += 1;
        } else {
            debugln!("adding to internal root");
            // find child that cotains the value
            let Some((median, right)) = self.root.find_and_insert(value, &cmp, self.depth) else {
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
            self.root.children_mut()[0] =
                NonNull::new(Box::into_raw(Box::new(left)).cast::<Node<T, N>>());
            self.root.children_mut()[1] =
                NonNull::new(Box::into_raw(Box::new(right)).cast::<Node<T, N>>());
            self.root.inner.len = 1;

            self.depth += 1;

            test_assert!(self.root.children()[..self.root.inner.len]
                .iter()
                .all(|x| x.is_some()));
            test_assert!(self
                .root
                .children()
                .get(self.root.inner.len + 1)
                .and_then(|c| c.as_ref())
                .is_none());
        }
    }
}
impl<T: Clone, const N: usize> BTree<T, N> {
    fn values(&self) -> Vec<T> {
        if self.depth == 0 {
            self.root.inner.values().cloned().collect::<Vec<T>>()
        } else {
            self.root.values(self.depth)
        }
    }
}
impl<T: Clone, const N: usize> InternalNode<T, N> {
    fn values(&self, depth: usize) -> Vec<T> {
        test_assert!(depth > 0);
        let mut values: Vec<T> = Vec::new();
        for (child, value) in self.children()[..self.inner.len + 1]
            .iter()
            .zip(self.inner.values().map(Some).chain(std::iter::once(None)))
        {
            let child = child.expect("child is None");
            if depth > 1 {
                let child = unsafe { child.cast::<InternalNode<T, N>>().as_ref() };
                test_assert!(child.inner.internal, "child is internal");
                values.extend(child.values(depth - 1));
            } else {
                let child = unsafe { child.cast::<LeafNode<T, N>>().as_ref() };
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
impl<T, const N: usize> InternalNode<T, N> {
    /// Returns the children of this node as a contiguous slice of N+1 elements. This is a
    /// ergonomic workaround the fact that we currently cannot represent a array of N+1 elements.
    fn children(&self) -> &[Option<NonNull<Node<T, N>>>] {
        // SAFETY: child0 and childs are layouted in memory as a contiguous array of N+1 elements.
        unsafe {
            test_assert!(
                std::ptr::addr_of!(self.child0).add(1) == std::ptr::addr_of!(self.childs[0])
            );
            // Due to pointer provenance, we cannot get a pointer to a field, and expand to the
            // entire array. Instead we pick a pointer to the struct, and shunk it to the array.
            let struct_ptr = std::ptr::addr_of!(*self);
            let offset = std::ptr::addr_of!(self.child0).byte_offset_from(struct_ptr);
            let array_ptr = struct_ptr.byte_offset(offset) as *const Option<NonNull<Node<T, N>>>;
            std::slice::from_raw_parts(array_ptr, N + 1)
        }
    }

    /// Returns the children of this node as a contiguous mutable slice of N+1 elements. This is a
    /// ergonomic workaround the fact that we currently cannot represent a array of N+1 elements.
    fn children_mut(&mut self) -> &mut [Option<NonNull<Node<T, N>>>] {
        // SAFETY: child0 and childs are layouted in memory as a contiguous array of N+1 elements.
        unsafe {
            test_assert!(
                std::ptr::addr_of!(self.child0).add(1) == std::ptr::addr_of!(self.childs[0])
            );
            // Due to pointer provenance, we cannot get a pointer to a field, and expand to the
            // entire array. Instead we pick a pointer to the struct, and shunk it to the array.
            let struct_ptr = std::ptr::addr_of_mut!(*self);
            let offset = std::ptr::addr_of!(self.child0).byte_offset_from(struct_ptr);
            let array_ptr = struct_ptr.byte_offset(offset) as *mut Option<NonNull<Node<T, N>>>;
            std::slice::from_raw_parts_mut(array_ptr, N + 1)
        }
    }
}

/// A node is either a Leaf or an InternalNode. But a pointer to a InternalNode<T> is a valid
/// pointer to a LeafNode<T>.
#[repr(transparent)]
#[derive(Debug)]
struct Node<T, const N: usize>(LeafNode<T, N>);

struct LeafNode<T: Sized, const N: usize> {
    values: [MaybeUninit<T>; N],
    len: usize,
    #[cfg(debug_assertions)]
    internal: bool,
}
impl<T: Debug, const N: usize> LeafNode<T, N> {
    fn new() -> Self {
        LeafNode {
            values: [(); N].map(|_| MaybeUninit::uninit()),
            len: 0,
            #[cfg(debug_assertions)]
            internal: false,
        }
    }

    /// Inserts a key into the node. If the node is full, the node is splited, in that case self
    /// becomes the left node and the median and the right node are returned.
    fn insert<F: Fn(&T, &T) -> Ordering>(&mut self, value: T, cmp: &F) -> Option<(T, Self)> {
        test_assert!(!self.internal);

        if self.len == N {
            let (median, right) = self.split_at_median(value, cmp);

            debugln!("leaf splited {:?} {:?} {:?}", self, median, right);

            test_assert!(!self.internal);
            test_assert!(!right.internal);

            return Some((median, right));
        }
        let mut i = self.len;
        // SAFETY: All elements with index < self.len are initialized. This invariant is valid at
        // initialization, with self.len = 0, and is maintained by this method.
        unsafe {
            while i > 0 && cmp(self.values[i - 1].assume_init_ref(), &value).is_gt() {
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

    fn split_at_median<F: Fn(&T, &T) -> Ordering>(
        &mut self,
        value: T,
        cmp: &F,
    ) -> (T, LeafNode<T, N>) {
        test_assert!(self.len == N);

        let mut right = LeafNode::<T, N>::new();
        // let median;

        // check if we need to insert the value in the self or right node
        let middle = N / 2;
        // SAFETY: self is fully initialized
        if cmp(&value, unsafe { self.values[middle].assume_init_ref() }).is_lt() {
            // we need to insert the value in the self node

            // move the right half of the keys to the right node
            //
            // SAFETY: all elements in self.values are initialized. The elements in
            // middle..N are being moved to right.values[0..CAPACITY - middle]. The length
            // of both nodes are updated, according to the number of elements moved.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.values.as_ptr().add(middle),
                    right.values.as_mut_ptr(),
                    N - middle,
                );
                self.len = middle;
                right.len = N - middle;
            }

            // insert the value in the self node
            self.insert(value, cmp).test_assert_none();

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
                    N - (middle + 1),
                );
                right.len = N - middle - 1;
            }

            // insert the value in the right node
            right.insert(value, cmp).test_assert_none();

            (median, right)
        }
    }
}
impl<T, const N: usize> LeafNode<T, N> {
    fn values(&self) -> impl Iterator<Item = &T> {
        // SAFETY: all indices smaller than self.len are initialized
        self.values[..self.len]
            .iter()
            .map(|v| unsafe { v.assume_init_ref() })
    }
}

#[repr(C)]
struct InternalNode<T, const N: usize> {
    inner: LeafNode<T, N>,
    /// The first child of this node.
    ///
    /// It could not be put in the array due to limitations to const
    /// generics in Rust.
    child0: Option<NonNull<Node<T, N>>>,
    /// The last N children of this node, the first child is `child0`.
    childs: [Option<NonNull<Node<T, N>>>; N],
}
impl<T: Debug, const N: usize> InternalNode<T, N> {
    fn new() -> Self {
        InternalNode {
            inner: LeafNode {
                #[cfg(debug_assertions)]
                internal: true,
                ..LeafNode::new()
            },
            child0: None,
            childs: [None; N],
        }
    }

    fn find_and_insert<F: Fn(&T, &T) -> Ordering>(
        &mut self,
        value: T,
        cmp: &F,
        depth: usize,
    ) -> Option<(T, Self)> {
        // find which child to insert the value
        let child_idx = 'find_child: {
            for (i, item) in self.inner.values().enumerate() {
                if cmp(&value, item).is_lt() {
                    break 'find_child i;
                }
            }
            self.inner.len
        };

        // insert the value in the child
        let child = self.children()[child_idx].expect("child is None");
        let (median, right) = if depth == 1 {
            // our child is a leaf
            let child = unsafe { child.cast::<LeafNode<T, N>>().as_mut() };
            test_assert!(!child.internal, "child is leaf");
            let (median, right) = child.insert(value, cmp)?;
            let right = NonNull::new(Box::into_raw(Box::new(right)).cast::<Node<T, N>>()).unwrap();
            (median, right)
        } else {
            // our child is an internal node
            let child: &mut InternalNode<T, N> =
                unsafe { child.cast::<InternalNode<T, N>>().as_mut() };
            test_assert!(child.inner.internal, "child is internal");
            let (median, right) = child.find_and_insert(value, cmp, depth - 1)?;
            let right = NonNull::new(Box::into_raw(Box::new(right)).cast::<Node<T, N>>()).unwrap();
            (median, right)
        };

        debugln!(
            "child splitted [{:?}] {:?} {:?}",
            child_idx,
            median,
            unsafe { right.as_ref() }
        );

        // self.insert(median, right)

        let Some((median, right)) = self.insert(median, right, cmp) else {
            return None;
        };

        // use std::fmt;
        // pub struct Fmt<F>(pub F)
        // where
        //     F: Fn(&mut fmt::Formatter) -> fmt::Result;
        //
        // impl<F> fmt::Debug for Fmt<F>
        // where
        //     F: Fn(&mut fmt::Formatter) -> fmt::Result,
        // {
        //     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        //         (self.0)(f)
        //     }
        // }
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
    fn insert<F: Fn(&T, &T) -> Ordering>(
        &mut self,
        value: T,
        child: NonNull<Node<T, N>>,
        cmp: &F,
    ) -> Option<(T, Self)> {
        debugln!("inserting {:?} {:?}", value, unsafe { child.as_ref() });

        if self.inner.len == N {
            let (median, rigth) = self.split_at_median(value, child, cmp);
            debugln!("internal splited {:?} {:?} {:?}", self, median, rigth);

            return Some((median, rigth));
        }
        let mut i = self.inner.len;
        // SAFETY: All elements with index < self.len are initialized. This invariant is valid at
        // initialization, with self.len = 0, and is maintained by this method.
        unsafe {
            while i > 0 && cmp(self.inner.values[i - 1].assume_init_ref(), &value).is_gt() {
                std::ptr::copy_nonoverlapping(
                    self.inner.values[i - 1].as_ptr(),
                    self.inner.values[i].as_mut_ptr(),
                    1,
                );
                i -= 1;
            }
        }

        // move the child pointers
        {
            let len = self.inner.len;
            self.children_mut().copy_within(i + 1..len + 1, i + 2);
        }

        self.inner.values[i].write(value);
        self.children_mut()[i + 1] = Some(child);
        self.inner.len += 1;

        test_assert!(
            self.inner.len >= N / 2,
            "invariant broken {:?} < {:?}",
            self.inner.len,
            N / 2
        );
        test_assert!(self.children()[..self.inner.len + 1]
            .iter()
            .all(|x| x.is_some()));

        None
    }

    fn split_at_median<F: Fn(&T, &T) -> Ordering>(
        &mut self,
        value: T,
        child: NonNull<Node<T, N>>,
        cmp: &F,
    ) -> (T, InternalNode<T, N>) {
        test_assert!(self.inner.len == N);

        let mut right = InternalNode::<T, N>::new();
        // let median;

        // check if we need to insert the value in the self or right node
        let middle = N / 2;
        // SAFETY: self is fully initialized
        let (median, right) = if unsafe {
            cmp(&value, self.inner.values[middle].assume_init_ref()).is_lt()
        } {
            // we need to insert the value in the self node

            // SAFETY: all elements in self.values are initialized. The elements in
            // self.values[middle..N] are being moved to right.values[0..CAPACITY - middle].
            // The elements in self.children[middle+1..N+1] are being moved to
            // right.children[1..N - middle + 1]. The length of both nodes are updated,
            // according to the number of elements moved.
            // After this, right.children[0] is still null.
            unsafe {
                // copy the right half of the keys to the right node
                std::ptr::copy_nonoverlapping(
                    self.inner.values.as_ptr().add(middle),
                    right.inner.values.as_mut_ptr(),
                    N - middle,
                );
                right.inner.len = N - middle;
                self.inner.len = middle;

                // copy children pointers
                {
                    let len = right.inner.len;
                    right.children_mut()[1..len + 1]
                        .copy_from_slice(&self.children()[middle + 1..]);
                }
            }

            // insert the value in the self node
            self.insert(value, child, cmp).test_assert_none();

            // SAFETY: pop the median out of self, updating the length of self accordingly, and
            // moving the popped child pointer to right.children[0].
            let median = unsafe {
                self.inner.len = middle;
                let median = self.inner.values[middle].assume_init_read();
                right.children_mut()[0] = self.children()[middle + 1];
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
                    N - (middle + 1),
                );
                right.inner.len = N - middle - 1;
            }

            // copy children pointers
            {
                let len = right.inner.len;
                right.children_mut()[..len + 1].copy_from_slice(&self.children()[middle + 1..]);
            }

            // insert the value in the right node
            let mut i = right.inner.len;
            // SAFETY: All elements with index < self.len are initialized. This invariant is valid at
            // initialization, with self.len = 0, and is maintained by this method.
            unsafe {
                while i > 0 && cmp(right.inner.values[i - 1].assume_init_ref(), &value).is_gt() {
                    std::ptr::copy_nonoverlapping(
                        right.inner.values[i - 1].as_ptr(),
                        right.inner.values[i].as_mut_ptr(),
                        1,
                    );
                    right.children_mut()[i + 1] = right.children()[i];
                    i -= 1;
                }
            }
            right.inner.values[i].write(value);
            right.children_mut()[i + 1] = Some(child);
            right.inner.len += 1;

            (median, right)
        };

        test_assert!(self.children()[..self.inner.len + 1]
            .iter()
            .all(|x| x.is_some()));
        test_assert!(right.children()[..right.inner.len + 1]
            .iter()
            .all(|x| x.is_some()));

        (median, right)
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
}
