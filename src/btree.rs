use super::debugln;
use std::cmp::Ordering;
use std::{fmt::Debug, mem::MaybeUninit, ptr::NonNull};

macro_rules! test_assert {
    ($cond:expr) => {
        #[cfg(all(debug_assertions, not(coverage)))]
        {
            assert!($cond);
        }
    };
    ($cond:expr, $($arg:tt)*) => {
        #[cfg(all(debug_assertions, not(coverage)))]
        {
            assert!($cond, $($arg)*);
        }
    };
}

mod debug;

#[cfg(test)]
mod test;

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
    /// The root of the tree, is either a LeafNode or an InternalNode, if height is 0 or not.
    root: InternalNode<T, N>,
    /// The height of the tree. A tree where the root is a leaf has height 0.
    height: usize,
}
impl<T: Debug, const N: usize> BTree<T, N> {
    pub fn new() -> Self {
        BTree {
            root: InternalNode {
                inner: LeafNode::new(),
                child0: None,
                childs: [None; N],
            },
            height: 0,
        }
    }

    pub fn insert<F: Fn(&T, &T) -> Ordering>(&mut self, value: T, cmp: F) {
        debugln!("adding {:?}", value);
        if self.height == 0 {
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
            self.height += 1;
        } else {
            debugln!("adding to internal root");
            // find child that cotains the value
            let Some((median, right)) = self.root.find_and_insert(value, &cmp, self.height) else {
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

            self.height += 1;

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

    pub fn remove<K: Debug + Copy, F: Fn(K, &T) -> Ordering>(
        &mut self,
        value: K,
        cmp: F,
    ) -> Option<T> {
        if self.height == 0 {
            debugln!("removing {:?} from leaf root", value);
            self.root.inner.remove(value, cmp)
        } else {
            debugln!("removing {:?} from internal root", value);
            let v = self.root.remove(value, cmp, self.height);

            // if the root has only one child, we can replace the root with a LeafNode.
            if self.root.inner.len == 0 {
                debugln!("root has only one child");
                let child = self.root.children_mut()[0].unwrap();
                if self.height > 1 {
                    self.root =
                        *unsafe { Box::from_raw(child.as_ptr().cast::<InternalNode<T, N>>()) };
                } else {
                    self.root.inner =
                        *unsafe { Box::from_raw(child.as_ptr().cast::<LeafNode<T, N>>()) };
                }
                self.height -= 1;
                debugln!("new root {:?}", self);
            }

            v
        }
    }
}
impl<T: Clone, const N: usize> BTree<T, N> {
    fn values(&self) -> Vec<T> {
        if self.height == 0 {
            self.root.inner.values().cloned().collect::<Vec<T>>()
        } else {
            self.root.values(self.height)
        }
    }
}
impl<T: Clone, const N: usize> InternalNode<T, N> {
    fn values(&self, height: usize) -> Vec<T> {
        test_assert!(height > 0);
        test_assert!(self.inner.internal, "node is internal");
        let mut values: Vec<T> = Vec::new();
        for (child, value) in self.children()[..self.inner.len + 1]
            .iter()
            .zip(self.inner.values().map(Some).chain(std::iter::once(None)))
        {
            let child = child.expect("child is None");
            if height > 1 {
                let child = unsafe { child.cast::<InternalNode<T, N>>().as_ref() };
                test_assert!(child.inner.internal, "child is internal");
                values.extend(child.values(height - 1));
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
impl<T, const N: usize> Node<T, N> {
    fn len(&self) -> usize {
        self.0.len
    }
}

struct LeafNode<T: Sized, const N: usize> {
    values: [MaybeUninit<T>; N],
    len: usize,
    #[cfg(debug_assertions)]
    internal: bool,
}
impl<T: Debug, const N: usize> LeafNode<T, N> {
    const MIN_VALUES: usize = N / 2;

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
            while i > 0 && cmp(&value, self.values[i - 1].assume_init_ref()).is_lt() {
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
        let middle = Self::MIN_VALUES;
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

    fn remove<K: Debug + Copy, F: Fn(K, &T) -> Ordering>(&mut self, value: K, cmp: F) -> Option<T> {
        // find the index of the value, if any
        let idx = 'search: {
            for (i, v) in self.values().enumerate() {
                if cmp(value, v).is_eq() {
                    break 'search i;
                }
            }

            debugln!(
                "could not find value {:?}: {:?}",
                value,
                self.values().collect::<Vec<_>>()
            );

            return None;
        };

        // remove the value from self.values
        debugln!("removing {:?} at {:?}", value, idx);

        let value = self.remove_at_idx(idx);

        Some(value)
    }

    fn remove_at_idx(&mut self, idx: usize) -> T {
        let value;
        // SAFETY: all elements with index < self.len are initialized. `self.values[i]` is
        // moved out of the array, and the rest of the array is shifted to the left. The
        // length of the array is updated accordingly.
        unsafe {
            value = self.values[idx].assume_init_read();
            let values_ptr = self.values.as_mut_ptr();
            std::ptr::copy(
                values_ptr.add(idx + 1),
                values_ptr.add(idx),
                self.len - (idx + 1),
            );
            self.len -= 1;
        }
        value
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
impl<T, const N: usize> InternalNode<T, N> {
    const MIN_VALUES: usize = N / 2;
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
        height: usize,
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
        let (median, right) = if height == 1 {
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
            let (median, right) = child.find_and_insert(value, cmp, height - 1)?;
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
        //     Fmt(|f| self.format(f, height - 1)),
        //     self,
        //     Fmt(|f| right.format(f, height))
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
            while i > 0 && cmp(&value, self.inner.values[i - 1].assume_init_ref()).is_lt() {
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
            self.inner.len >= Self::MIN_VALUES,
            "invariant broken {:?} < {:?}",
            self.inner.len,
            Self::MIN_VALUES
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
        let middle = Self::MIN_VALUES;
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
                while i > 0 && cmp(&value, right.inner.values[i - 1].assume_init_ref()).is_lt() {
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

    fn remove<K: Debug + Copy, F: Fn(K, &T) -> Ordering>(
        &mut self,
        value: K,
        cmp: F,
        height: usize,
    ) -> Option<T> {
        // find the index of the value, or try to remove from a child.
        let idx = 'search: {
            let mut idx = self.inner.len;
            for (i, v) in self.inner.values().enumerate() {
                debugln!("comparing {:?} {:?}", value, v);
                let ord = cmp(value, v);
                if ord.is_eq() {
                    break 'search i;
                }
                if ord.is_lt() {
                    debugln!("value is greater");
                    idx = i;
                    break;
                }
            }

            debugln!("value not found, trying to remove from child {:?}", idx);

            // value not found, remove from children.
            let child = self.children_mut()[idx].unwrap();
            let (removed, len) = if height > 1 {
                // SAFETY: idx <= self.inner.len, so self.children[idx] is a valid child.
                let child = unsafe { child.cast::<InternalNode<T, N>>().as_mut() };
                test_assert!(child.inner.internal, "child is internal");
                (child.remove(value, cmp, height - 1), child.inner.len)
            } else {
                // SAFETY: idx <= self.inner.len, so self.children[idx] is a valid child.
                let child = unsafe { child.cast::<LeafNode<T, N>>().as_mut() };
                test_assert!(!child.internal, "child is leaf");
                (child.remove(value, cmp), child.len)
            };

            debugln!("removed from child {:?} {:?}", removed, len);

            if len < Self::MIN_VALUES {
                self.rebalance(idx, height);
            }

            return removed;
        };
        debugln!("removing {:?} at {:?}", value, idx);

        let value = self.remove_at_idx(height, idx);

        Some(value)
    }

    /// Rebalance the child at index `idx`.
    fn rebalance(&mut self, idx: usize, height: usize) {
        debugln!("rebalacing child {:?} at height {:?}", idx, height);

        // if the right sibling has more than N/2 elements, rotate left
        if idx < self.inner.len
            && unsafe { self.children()[idx + 1].unwrap().as_ref().len() } > Self::MIN_VALUES
        {
            debugln!("rotating left");
            let left_ptr = self.children()[idx].unwrap();
            let right_ptr = self.children()[idx + 1].unwrap();

            // SAFETY: idx < self.inner.len, so self.children[idx] and self.children[idx + 1] are
            // valid children. All nodes are valid LeaftNodes, so it is safe to cast them to
            // LeafNode, regardless of the height.
            let left = unsafe { left_ptr.cast::<LeafNode<T, N>>().as_mut() };
            let right = unsafe { right_ptr.cast::<LeafNode<T, N>>().as_mut() };

            // move the separator the the left child

            // SAFETY: left is not full, right has at least N/2 elements. We move the separator to
            // the left child, and move the first element of right to the separator position. And
            // we shift the elements of right to the left. We update the lengths accordingly.
            unsafe {
                test_assert!(left.len < N);
                left.values[left.len].write(self.inner.values[idx].assume_init_read());
                left.len += 1;

                self.inner.values[idx].write(right.values[0].assume_init_read());

                let rvalues_ptr = right.values.as_mut_ptr();
                std::ptr::copy(rvalues_ptr.add(1), rvalues_ptr, right.len - 1);
                right.len -= 1;
            }

            // move the first child of right to the last child of left.
            if height > 1 {
                // SAFETY: height > 1, so the children are InternalNodes.
                let left = unsafe { left_ptr.cast::<InternalNode<T, N>>().as_mut() };
                let right = unsafe { right_ptr.cast::<InternalNode<T, N>>().as_mut() };
                test_assert!(left.inner.internal, "left is internal");
                test_assert!(left.inner.internal, "right is internal");

                let llen = left.inner.len;
                let rlen = right.inner.len;
                left.children_mut()[llen] = right.children_mut()[0];
                right.children_mut().copy_within(1..rlen + 2, 0);
            }

            return;
        }

        // if the left sibling has more than N/2 elements, rotate right
        if idx > 0 && unsafe { self.children()[idx - 1].unwrap().as_ref().len() } > Self::MIN_VALUES
        {
            debugln!("rotating right");
            let left_ptr = self.children()[idx - 1].unwrap();
            let right_ptr = self.children()[idx].unwrap();

            // SAFETY: idx > 0, so self.children[idx] and self.children[idx - 1] are
            // valid children. All nodes are valid LeaftNodes, so it is safe to cast them to
            // LeafNode, regardless of the height.
            let left = unsafe { left_ptr.cast::<LeafNode<T, N>>().as_mut() };
            let right = unsafe { right_ptr.cast::<LeafNode<T, N>>().as_mut() };

            // move the separator the the right child

            // SAFETY: right is not full, left has at least N/2 elements. We move the separator to
            // the right child, and move the last element of left to the separator position. And
            // we shift the elements of left to the right. We update the lengths accordingly.
            unsafe {
                test_assert!(right.len < N);
                let values_ptr = right.values.as_mut_ptr();
                std::ptr::copy(values_ptr, values_ptr.add(1), right.len);
                right.len += 1;

                right.values[0].write(self.inner.values[idx - 1].assume_init_read());
                left.len -= 1;

                self.inner.values[idx - 1].write(left.values[left.len].assume_init_read());
            }

            // move the last child of left to the first child of right.
            if height > 1 {
                // SAFETY: height > 1, so the children are InternalNodes.
                let left = unsafe { left_ptr.cast::<InternalNode<T, N>>().as_mut() };
                let right = unsafe { right_ptr.cast::<InternalNode<T, N>>().as_mut() };

                let rlen = left.inner.len;
                let llen = left.inner.len;
                right.children_mut().copy_within(0..rlen, 1);
                right.children_mut()[0] = left.children_mut()[llen + 1];
            }

            return;
        }

        // merge with a sibling

        let i;
        if idx < self.inner.len {
            i = idx;
        } else if idx > 0 {
            i = idx - 1;
        } else {
            unreachable!();
        }

        debugln!("merge {:?} and {:?}", i, i + 1);

        let left_ptr = self.children()[i].unwrap();
        let right_ptr = self.children()[i + 1].unwrap();

        // SAFETY: we check valid of i above. All nodes are valid LeaftNodes, so it is safe to cast
        // them to LeafNode, regardless of the height.
        let left = unsafe { left_ptr.cast::<LeafNode<T, N>>().as_mut() };
        let right = unsafe { right_ptr.cast::<LeafNode<T, N>>().as_mut() };

        // move the separator to the left child, and move all elements of right to left

        // SAFETY: left has less than N/2 elements, right has less than N/2 elements. So we can
        // move all elements to left.
        unsafe {
            test_assert!(
                left.len + 1 + right.len <= N,
                "{} {} {}",
                left.len,
                right.len,
                N
            );
            left.values[left.len].write(self.inner.values[i].assume_init_read());
            std::ptr::copy_nonoverlapping(
                right.values.as_ptr(),
                left.values.as_mut_ptr().add(left.len + 1),
                right.len,
            );

            let llen = left.len;
            let rlen = right.len;
            left.len = left.len + 1 + right.len;

            if height > 1 {
                let left = left_ptr.cast::<InternalNode<T, N>>().as_mut();
                let right = right_ptr.cast::<InternalNode<T, N>>().as_mut();

                left.children_mut()[llen + 1..llen + 1 + rlen + 1]
                    .copy_from_slice(&right.children()[..rlen + 1]);
            }
        }

        debugln!("left {:?}", unsafe {
            left_ptr.cast::<LeafNode<T, N>>().as_mut()
        });

        // remove the separator and right child, and shift the rest of the children

        // SAFETY: i is a valid index, and the length of the array is updated accordingly.
        unsafe {
            let len = self.inner.len;
            let values_ptr = self.inner.values.as_mut_ptr();
            std::ptr::copy(values_ptr.add(i + 1), values_ptr.add(i), len - (i + 1));
            self.children_mut().copy_within(i + 2..len + 1, i + 1);
            self.inner.len -= 1;
        }

        // drop right allocation
        unsafe {
            if height > 1 {
                let _ = Box::from_raw(right_ptr.cast::<InternalNode<T, N>>().as_ptr());
            } else {
                let _ = Box::from_raw(right_ptr.cast::<LeafNode<T, N>>().as_ptr());
            }
        }

        debugln!("{:?}", self);
    }

    fn remove_at_idx(&mut self, height: usize, idx: usize) -> T {
        // replace the value with the biggest value of the left child's subtree.
        let value;
        let predecessor;
        let len;

        if height > 1 {
            let left_child = unsafe {
                self.children_mut()[idx]
                    .expect("child is None")
                    .cast::<InternalNode<T, N>>()
                    .as_mut()
            };

            predecessor = left_child.remove_max_value(height - 1);

            len = left_child.inner.len;
        } else {
            // SAFETY: `idx` is a index of a valid element in self.values, so the same index is
            // valid in self.children. The child is a leaf node (height==1), so it is safe to cast
            // it to LeafNode.
            let left_child = unsafe {
                self.children_mut()[idx]
                    .expect("child is None")
                    .cast::<LeafNode<T, N>>()
                    .as_mut()
            };

            debugln!("removing from leaf {:?}", left_child);

            //  pop from left_chid

            // SAFETY: the last element in left_child is initialized, and is moved out, and the
            // length of the array is updated accordingly.
            unsafe {
                left_child.len -= 1;
                predecessor = left_child.values[left_child.len].assume_init_read();
            }

            len = left_child.len;
        }
        value = std::mem::replace(
            // SAFETY: idx is a valid index.
            unsafe { self.inner.values[idx].assume_init_mut() },
            predecessor,
        );

        if len < Self::MIN_VALUES {
            self.rebalance(idx, height);
        }

        value
    }

    fn remove_max_value(&mut self, height: usize) -> T {
        debugln!("removing max value at height {:?}", height);
        if height > 1 {
            let left_child = unsafe {
                let len = self.inner.len;
                self.children_mut()[len]
                    .expect("child is None")
                    .cast::<InternalNode<T, N>>()
                    .as_mut()
            };

            let value = left_child.remove_max_value(height - 1);

            if left_child.inner.len < Self::MIN_VALUES {
                self.rebalance(self.inner.len, height);
            }

            value
        } else {
            // SAFETY: `idx` is a index of a valid element in self.values, so the same index is
            // valid in self.children. The child is a leaf node (height==1), so it is safe to cast
            // it to LeafNode.
            let left_child = unsafe {
                let len = self.inner.len;
                self.children_mut()[len]
                    .expect("child is None")
                    .cast::<LeafNode<T, N>>()
                    .as_mut()
            };

            debugln!("removing from leaf {:?}", left_child);

            //  pop from left_chid

            // SAFETY: the last element in left_child is initialized, and is moved out, and the
            // length of the array is updated accordingly.
            let value = unsafe {
                left_child.len -= 1;
                left_child.values[left_child.len].assume_init_read()
            };

            if left_child.len < Self::MIN_VALUES {
                self.rebalance(self.inner.len, height);
            }

            value
        }
    }
}

impl<T, const N: usize> Drop for BTree<T, N> {
    fn drop(&mut self) {
        if self.height == 0 {
            unsafe {
                self.root.inner.drop_in_place();
            }
        } else {
            unsafe {
                self.root.drop_in_place(self.height);
            }
        }
    }
}
impl<T, const N: usize> LeafNode<T, N> {
    /// SAFETY: should only be called once, on BTree drop.
    unsafe fn drop_in_place(&mut self) {
        // SAFETY: all elements with index < self.len are initialized
        unsafe {
            for i in 0..self.len {
                std::ptr::drop_in_place(self.values[i].as_mut_ptr());
            }
        }
    }
}
impl<T, const N: usize> InternalNode<T, N> {
    /// SAFETY: should only be called once, on BTree drop.
    unsafe fn drop_in_place(&mut self, height: usize) {
        let len = self.inner.len + 1;
        for child in &mut self.children_mut()[..len] {
            let child = child.expect("child is None");
            if height > 1 {
                // SAFETY: if height > 1, the children are InternalNodes.
                unsafe {
                    debugln!("drop internal {:?}", child);
                    let mut child = Box::from_raw(child.as_ptr().cast::<InternalNode<T, N>>());
                    (*child).drop_in_place(height - 1);
                }
            } else {
                // SAFETY: if height == 1, the children are LeaftNodes.
                unsafe {
                    debugln!("drop leaf {:?}", child);
                    let mut child = Box::from_raw(child.as_ptr().cast::<LeafNode<T, N>>());
                    (*child).drop_in_place();
                }
            }
        }
        for value in &mut self.inner.values[..self.inner.len] {
            std::ptr::drop_in_place(value.as_mut_ptr());
        }
    }
}
