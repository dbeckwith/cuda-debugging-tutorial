extern crate cuda_debugging_tutorial;

use std::env;
use std::path::PathBuf;

fn main() {
    let mut args = env::args();
    args.next().unwrap();
    let input_path = PathBuf::from(args.next().expect("Input path not specified"));
    let output_path = PathBuf::from(args.next().expect("Output path not specified"));

    if let Err(ref e) = cuda_debugging_tutorial::run_blur(&input_path, &output_path) {
        eprintln!("error: {}", e);

        for e in e.iter().skip(1) {
            eprintln!("caused by: {}", e);
        }

        if let Some(backtrace) = e.backtrace() {
            eprintln!("backtrace: {:?}", backtrace);
        }

        ::std::process::exit(1);
    }
}
