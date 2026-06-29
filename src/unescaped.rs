//! A wrapper for strings that have escape characters in them
use std::fmt::{self, Display, Formatter};
use std::iter::{FusedIterator, Peekable};
use std::str::Chars;

/// A string with backslash escapes in it
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct EscapedStr {
    escaped: str,
}

impl EscapedStr {
    pub(crate) fn new(escaped: &str) -> &Self {
        debug_assert!(
            escaped
                .match_indices('"')
                .all(|(idx, _)| escaped[..idx].ends_with('\\')),
            "EscapedStr must not contain an unescaped quote, got {escaped:?}"
        );
        // SAFETY: `EscapedStr` is `#[repr(transparent)]` over `str`, so a `&str` and a
        // `&EscapedStr` share the same layout.
        unsafe { &*(escaped as *const str as *const EscapedStr) }
    }

    /// The string in its original, escaped form
    pub fn escape(&self) -> &str {
        &self.escaped
    }

    /// Get an iterator over the true characters
    pub fn unescape(&self) -> Unescaped<'_> {
        Unescaped {
            chars: self.escaped.chars().peekable(),
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
    chars: Peekable<Chars<'a>>,
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
        let chr = self.chars.next()?;
        if let ('\\', Some(&'"')) = (chr, self.chars.peek()) {
            self.chars.next()
        } else {
            Some(chr)
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
        assert_eq!(escaped.to_string(), "air \" quote");
    }

    #[test]
    fn unescapes_only_quotes() {
        assert_eq!(EscapedStr::new(r#"a\"b"#).to_string(), r#"a"b"#);
        assert_eq!(EscapedStr::new(r"a\b").to_string(), r"a\b");
        assert_eq!(EscapedStr::new(r"a\nb").to_string(), r"a\nb");
        assert_eq!(EscapedStr::new(r"a\\b").to_string(), r"a\\b");
    }
}
