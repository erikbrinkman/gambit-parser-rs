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

Remarks
-------

Gambit's reader duplicates runs of backslashes when reading a quoted label;
this parser does not reproduce that bug.

To Do
-----

Ultimately this represents a data model that could be modified and serialized,
but that's not implemented yet. The current version keeps a reference to the
underlying file bytes, to implement a full data model there should be an owned
version of the `ExtensiveFormGame` that supports full serialization and
modification.
