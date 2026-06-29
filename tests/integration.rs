use gambit_parser::ExtensiveFormGame;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::{self, Read};

/// Labels follow Gambit's escaping: `\"` is a literal quote and every other backslash is kept
#[test]
fn parses_escaped_labels() {
    let game_str = r#"EFG 2 R "a \"quoted\" name" { "p\one" "two" } t "" 1 { 1 2 }"#;
    let game: ExtensiveFormGame<'_> = game_str.try_into().unwrap();

    // Display gives the real text, `\"` collapsed to `"`
    assert_eq!(game.name().to_string(), r#"a "quoted" name"#);
    // escape() gives the preserved form, escapes intact
    assert_eq!(game.name().escape(), r#"a \"quoted\" name"#);
    // a backslash before a normal character is preserved verbatim
    assert_eq!(game.player_names()[0].to_string(), r"p\one");
    assert_eq!(game.player_names()[1].to_string(), "two");
}

/// Every label here carries an escaped quote, so serializing any of them unescaped would corrupt
/// the re-parse. Comparing the parsed trees guards against exactly that.
#[test]
fn round_trips_escaped_labels() {
    let game_str = r#"EFG 2 R "g\"name" { "p\"1" "p\"2" } "comm\"ent"
c "ch\"node" 1 "ch\"infoset" { "a\"1" 1/2 "a\"2" 1/2 } 0
p "pl\"1" 1 1 "in\"1" { "H\"" "L\"" } 0
t "t\"a" 1 "ou\"1" { 10 2 }
t "t\"b" 2 "ou\"2" { 0 10 }
p "pl\"2" 2 1 "in\"2" { "h\"" "l\"" } 0
t "t\"c" 3 "ou\"3" { 2 4 }
t "t\"d" 4 "ou\"4" { 4 0 }
"#;
    let parsed: ExtensiveFormGame<'_> = game_str.try_into().unwrap();
    let written = parsed.to_string();
    let reparsed: ExtensiveFormGame<'_> = written.as_str().try_into().unwrap();
    assert_eq!(parsed, reparsed);
}

/// Test that we can read each example efg gambit file, and that when writing and rereading them we
/// get the same result
#[test]
fn test_pairity() -> io::Result<()> {
    let mut original = String::new();
    let mut formatted = String::new();
    for entry_result in fs::read_dir("resources")? {
        let entry = entry_result?;
        let raw_name = entry.file_name();
        let filename = raw_name.to_str().unwrap();
        if filename.ends_with(".efg") {
            original.clear();
            formatted.clear();
            File::open(entry.path())?.read_to_string(&mut original)?;
            let orig: ExtensiveFormGame<'_> = original.as_str().try_into().unwrap();
            write!(&mut formatted, "{}", orig).unwrap();
            let clone: ExtensiveFormGame<'_> = formatted.as_str().try_into().unwrap();
            assert_eq!(orig, clone);
        }
    }
    Ok(())
}
