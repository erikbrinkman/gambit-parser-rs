Gambit Parser
=============
[![crates.io](https://img.shields.io/crates/v/gambit-parser)](https://crates.io/crates/gambit-parser)
[![docs](https://docs.rs/gambit-parser/badge.svg)](https://docs.rs/gambit-parser)
[![license](https://img.shields.io/github/license/erikbrinkman/gambit-parser-rs)](LICENSE)
[![tests](https://github.com/erikbrinkman/gambit-parser-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/erikbrinkman/gambit-parser-rs/actions/workflows/rust.yml)

A rust parser for gambit [extensive form game
(`.efg`)](https://gambitproject.readthedocs.io/en/v16.0.2/formats.html) files.

Usage
-----

```rust
use gambit_parser::ExtensiveFormGame;
use std::fs::File;
use std::io::Read;

let mut buffer = String::new();
File::open("my path")?.read_to_string(&mut buffer)?;
let parsed: ExtensiveFormGame<'_> = buffer.as_str().try_into()?;
```

A parsed game serializes back to the `.efg` format via `Display`:

```rust
use gambit_parser::WriteMode;

// `to_string()` (plain `Display`) reproduces the input faithfully
let text = parsed.to_string();
// or pick how much of each shared infoset/outcome to repeat
let compact = parsed.display(WriteMode::Minimal).to_string();
let expanded = parsed.display(WriteMode::Exhaustive).to_string();
```

Remarks
-------

Gambit's reader duplicates runs of backslashes when reading a quoted label;
this parser does not reproduce that bug.

Chance-node probabilities are kept as written and are not required to sum to
one, matching Gambit (which performs no such check). A file that approximates a
distribution with rounded decimals — e.g. `0.333333` repeated three times,
summing to `0.999999` — parses fine; normalize the probabilities yourself if you
need an exact distribution.

This parser diverges from Gambit's self-delimiting lexer in a few ways that only
affect hand-written files; everything Gambit's own writer emits round-trips. It
requires whitespace between tokens, so `"x"1`, `t"" 1`, and an empty `{ }`
player list are rejected where Gambit accepts them, and it rejects trailing
content after the root subtree that Gambit ignores. Conversely, it compares
repeated outcome and probability definitions numerically rather than textually,
so `{ 1/2, 2 }` followed by `{ 0.5, 2 }` for the same id parses here but errors
in Gambit.

To Do
-----

Parsing and serialization are implemented, but the parsed `ExtensiveFormGame`
borrows the underlying file bytes and is read-only. A full data model would need
an owned version of `ExtensiveFormGame` that can be constructed and modified in
memory rather than only parsed from text; that's not implemented yet.
