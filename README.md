Gambit Parser
=============
[![crates.io](https://img.shields.io/crates/v/gambit-parser)](https://crates.io/crates/gambit-parser)
[![docs](https://img.shields.io/badge/docs-docs.rs-blue)](https://docs.rs/cfr/latest/gambit-parser/)
[![build](https://github.com/erikbrinkman/gambit-parser-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/erikbrinkman/gambit-parser-rs/actions/workflows/rust.yml)
[![license](https://img.shields.io/github/license/erikbrinkman/gambit-parser-rs)](LICENSE)

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
let parsed: ExtensiveFormGame<'_> = original.as_str().try_into()?;
```

Remarks
-------

The gambit spec says that the list of actions in chance and player nodes is
technically optional. For this to be optional, they would need to be defined
for the same infoset in the same file. Handling this case is slightly more
difficult and not well documented. Since I couldn't find any examples of a
file like this, this specific omission isn't handled.

To Do
-----

Ultimately this represents a data model that could be modified and serialized,
but that's not implemented yet. The current version keeps a reference to the
underlying file bytes, to implement a full data model there should be an owned
version of the `ExtensiveFormGame` that supports full serialization and
modification.
