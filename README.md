# How to Debug Rust CUDA Applications

## Project Layout

This crate demonstrates a simple way to set up a Rust application to use CUDA kernels to perform vectorized computation for parts of the application. Since the CUDA kernels need to be compiled separately from the main Rust application code, the project uses several sub-crates to allow re-use of code. The main crate contains the application code. The `kernel` crate contains the signatures of the CUDA kernels that will need to be run. The `common` crate contains any code that needs to be used by both the application crate and the `kernel` crate. An important feature of the `common` crate is that it can be compiled in a `#[no_std]` environment, since code running on a CUDA device needs to be able to run without the Rust standard library. If you still want the `common` crate to use `std` components while being used in the application crate, the `common` crate needs to be conditionally `#[no_std]`, which is implemented in this application with a [crate feature](https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section).

## Build Process

The application crate uses a [Cargo build script](https://doc.rust-lang.org/cargo/reference/build-scripts.html) ([`build.rs`](./build.rs)). Which will compile the `kernel` crate to CUDA PTX assembly before the application crate is built. The path to the produced PTX file is then provided to the application crate in an environment variable at build time. The application crate loads the text of this file at build time and embeds it in the binary using the [`include_str!` macro](https://doc.rust-lang.org/std/macro.include_str.html). At run time, the application gives this PTX code to the CUDA driver, which compiles it into an actual CUDA kernel which can then be run.

This process requires some tools beyond Cargo in order to run. Before building, make sure you run the [`install-build-tools.sh` script](./install-build-tools.sh). It will install [Xargo](https://github.com/japaric/xargo), [PTX Builder](https://github.com/denzp/rust-ptx-builder), and the LLVM development libraries if you're on a system with the `apt` package installer. If you don't have `apt` you can look at the script and install the package yourself. The script will only install the tools if they are not already present. If you have versions of Xargo or PTX Builder already installed that are not compatible, you may run into build errors. Check the script for the required versions.

## Running the Application

To run the demo application, first find an image (try it out with a particularly large one). Then run:

```
$ cargo run --release -- <blur_radius> path/to/input_image path/to/output_image
```

To produce a blurred image. The CUDA kernel is set up to perform a simple [box blur](https://en.wikipedia.org/wiki/Box_blur).

Here are the steps involved in this process:
* Load the image, converting to RGB if necessary.
* Find a suitable CUDA device on your system and initialize the CUDA driver.
* Load the CUDA kernel from the embedded PTX assembly code.
* Initialize memory buffers on the CUDA device to hold the image data.
* Copy the input image data from the host to the device.
* Partition the work needed to run the kernel (in this application each pixel is an item of work) into [CUDA blocks](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels).
* Run the kernel, which performs a box blur on each pixel.
* Copy the output image from the device to the host.
* Save the output image.

See the annotated source for more.
