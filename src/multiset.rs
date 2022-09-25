//! A simple wrapper for multisets
// FIXME  export this as it's own crate
// The exported crate should mimic std::collections Set and Hash, ideally also generic on numeric
// types, also thing about checked vs saturating for counts
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BTreeMultiSet<K>(BTreeMap<K, usize>);

impl<K: Ord> FromIterator<K> for BTreeMultiSet<K> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = K>,
    {
        let mut map = BTreeMap::new();
        for item in iter {
            *map.entry(item).or_insert(0) += 1
        }
        BTreeMultiSet(map)
    }
}
