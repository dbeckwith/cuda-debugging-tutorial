#![feature(abi_ptx, intrinsics)]
#![no_std]

extern crate common;
extern crate nvptx_builtins;

use common::{kernel, ImageInfo, Pixel};
use nvptx_builtins::*;

#[no_mangle]
pub extern "ptx-kernel" fn gaussian_blur(
    input_pixels: *const Pixel,
    output_pixels: *mut Pixel,
    info: *const ImageInfo,
    radius: f64,
) {
    let x = unsafe { block_idx_x() * block_dim_x() + thread_idx_x() } as usize;
    let y = unsafe { block_idx_y() * block_dim_y() + thread_idx_y() } as usize;

    kernel::gaussian_blur(input_pixels, output_pixels, info, radius, x, y);
}
