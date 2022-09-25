use gambit_parser::ExtensiveFormGame;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::{self, Read};

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
