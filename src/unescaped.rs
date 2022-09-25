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
pub struct EscapedStr {
    escaped: str,
}

impl EscapedStr {
    // NOTE this could panic if not called on a validated string
    pub(crate) fn new<S: AsRef<str> + ?Sized>(escaped: &S) -> &Self {
        unsafe { &*(escaped.as_ref() as *const str as *const EscapedStr) }
    }

    /// Access the underly buffer with escape sequences in it
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
        ((min + 1) / 2, max)
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
