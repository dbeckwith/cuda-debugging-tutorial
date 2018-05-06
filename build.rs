extern crate ptx_builder;

use ptx_builder::{builder::{BuildStatus, Builder, Profile},
                  error::Result,
                  reporter::BuildReporter};
use std::env;
use std::process::exit;

fn main() {
    if let Err(error) = build() {
        eprintln!("{}", BuildReporter::report(error));
        exit(1);
    }
}

fn build() -> Result<()> {
    let mut builder = Builder::new("kernel")?;
    builder.set_profile(match env::var("PROFILE") {
        Ok(s) => match s.as_ref() {
            "debug" => Profile::Debug,
            "release" => Profile::Release,
            _ => Profile::Release,
        },
        _ => Profile::Release,
    });

    match builder.build()? {
        BuildStatus::Success(output) => {
            // Provide the PTX Assembly location via env variable
            println!(
                "cargo:rustc-env=KERNEL_PTX_PATH={}",
                output.get_assembly_path().to_str().unwrap()
            );

            // Observe changes in kernel sources
            for path in output.source_files()? {
                println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
            }
        }

        BuildStatus::NotNeeded => {
            println!("cargo:rustc-env=KERNEL_PTX_PATH=/dev/null");
        }
    };

    Ok(())
}
