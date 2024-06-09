use super::{BTree, InternalNode, LeafNode};
use std::fmt::Debug;

impl<T: Debug, F, const N: usize> Debug for BTree<T, F, N> {
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

impl<T: Debug, const N: usize> Debug for InternalNode<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "~InternalNode(")?;
        self.inner.format(f)?;
        write!(f, ")")?;
        Ok(())
    }
}

impl<T: Debug, const N: usize> Debug for LeafNode<T, N> {
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

impl<T: Debug, const N: usize> InternalNode<T, N> {
    pub(crate) fn format(&self, f: &mut std::fmt::Formatter<'_>, depth: usize) -> std::fmt::Result {
        test_assert!(depth > 0);
        write!(f, "[").unwrap();
        for (child, value) in self.children()[..self.inner.len + 1]
            .iter()
            .zip(self.inner.values().map(Some).chain(std::iter::once(None)))
        {
            let child = child.expect("child is None");
            if depth > 1 {
                let child = unsafe { child.cast::<InternalNode<T, N>>().as_ref() };
                test_assert!(child.inner.internal, "child is internal");
                child.format(f, depth - 1)?;
            } else {
                let child = unsafe { child.cast::<LeafNode<T, N>>().as_ref() };
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

impl<T: Debug, const N: usize> LeafNode<T, N> {
    pub(crate) fn format(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
