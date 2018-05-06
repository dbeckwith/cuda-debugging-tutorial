#[macro_use]
extern crate error_chain;
extern crate common;
extern crate cuda;
extern crate image;

mod errors;

use errors::*;
use std::path::Path;

pub fn run_blur(input_path: &Path, output_path: &Path) -> Result<()> {
    Ok(())
}
