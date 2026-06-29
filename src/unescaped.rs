//! A wrapper for strings that have escape characters in them
use std::fmt::{self, Display, Formatter};
use std::iter::FusedIterator;
use std::str::Chars;

/// A string with backslash escapes in it
///
/// This wrapper allows referencing escaped strings in the source while preventing improper use.
/// Use [EscapedStr::as_raw_str] to access the underlying buffer, or the [Display] trait to get
/// owned versions. Use [EscapedStr::unescape] to access a [char] iter.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct EscapedStr {
    escaped: str,
}

impl EscapedStr {
    pub(crate) fn new<S: AsRef<str> + ?Sized>(escaped: &S) -> &Self {
        let escaped = escaped.as_ref();
        // SAFETY: `EscapedStr` is `#[repr(transparent)]` over `str`, so a `&str` and a
        // `&EscapedStr` share the same layout and this reference cast is sound.
        unsafe { &*(escaped as *const str as *const EscapedStr) }
    }

    /// Access the underlying buffer with escape sequences in it
    pub fn as_raw_str(&self) -> &str {
        &self.escaped
    }

    /// Get an iterator over the true characters
    pub fn unescape(&self) -> Unescaped<'_> {
        Unescaped {
            chars: self.escaped.chars(),
        }
    }
}

impl Display for EscapedStr {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(out, "{}", self.unescape())
    }
}

/// An iterator over the true characters of an [EscapedStr]
#[derive(Debug, Clone)]
pub struct Unescaped<'a> {
    chars: Chars<'a>,
}

impl<'a> Display for Unescaped<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        for chr in self.clone() {
            write!(out, "{}", chr)?;
        }
        Ok(())
    }
}

impl<'a> Iterator for Unescaped<'a> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        match self.chars.next() {
            Some('\\') => Some(self.chars.next().unwrap()),
            chr => chr,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self.chars.size_hint();
        (min.div_ceil(2), max)
    }
}

impl<'a> FusedIterator for Unescaped<'a> {}

#[cfg(test)]
mod tests {
    use super::EscapedStr;

    #[test]
    fn test_formatting() {
        let escaped = EscapedStr::new("air \\\" quote");
        let res = escaped.to_string();
        assert_eq!(res, "air \" quote");
    }
}
