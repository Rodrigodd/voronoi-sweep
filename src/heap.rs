use core::fmt;
use std::cmp::Ordering;

/// A min-heap, with a custom comparator.
pub struct Heap<T, F> {
    data: Vec<T>,
    cmp: F,
}
impl<T, F: Fn(&T, &T) -> Ordering> Heap<T, F> {
    pub fn new(cmp: F) -> Self {
        Heap {
            data: Vec::new(),
            cmp,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn push(&mut self, value: T) {
        let mut i = self.data.len();
        self.data.push(value);
        while i > 0 {
            let p = (i - 1) / 2;
            if (self.cmp)(&self.data[i], &self.data[p]) == Ordering::Less {
                self.data.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let ret = self.data.swap_remove(0);
        let mut i = 0;
        while 2 * i + 1 < self.data.len() {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut j = l;
            if r < self.data.len() && (self.cmp)(&self.data[r], &self.data[l]) == Ordering::Less {
                j = r;
            }
            if (self.cmp)(&self.data[j], &self.data[i]) == Ordering::Less {
                self.data.swap(i, j);
                i = j;
            } else {
                break;
            }
        }
        Some(ret)
    }

    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn remove(&mut self, f: impl Fn(&T) -> bool) -> Option<T> {
        let index = self.data.iter().position(f)?;
        Some(self.remove_at(index))
    }

    pub fn remove_at(&mut self, index: usize) -> T {
        assert!(index < self.data.len());
        let ret = self.data.swap_remove(index);
        if index == self.data.len() {
            return ret;
        }
        let mut i = index;
        while i > 0 {
            let p = (i - 1) / 2;
            if (self.cmp)(&self.data[i], &self.data[p]) == Ordering::Less {
                self.data.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
        while 2 * i + 1 < self.data.len() {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut j = l;
            if r < self.data.len() && (self.cmp)(&self.data[r], &self.data[l]) == Ordering::Less {
                j = r;
            }
            if (self.cmp)(&self.data[j], &self.data[i]) == Ordering::Less {
                self.data.swap(i, j);
                i = j;
            } else {
                break;
            }
        }
        ret
    }
}

impl<T: fmt::Debug, F> fmt::Debug for Heap<T, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_heap() {
        let mut heap = Heap::new(|a: &i32, b: &i32| a.cmp(b));
        assert_eq!(heap.pop(), None);
        heap.push(3);
        heap.push(2);
        heap.push(1);
        heap.push(4);
        assert_eq!(heap.len(), 4);
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.pop(), None);
        assert_eq!(heap.len(), 0);
    }

    proptest! {
        #[test]
        fn always_sort(mut items: Vec<u32>) {
            let mut heap = Heap::new(u32::cmp);

            for item in items.iter().copied() {
                heap.push(item);
            }

            let mut heap_order = vec![];
            while let Some(item) = heap.pop() {
                heap_order.push(item);
            }

            items.sort();

            assert_eq!(items, heap_order);
        }
    }
}
