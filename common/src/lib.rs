#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), feature(core_intrinsics))]

pub mod kernel;

#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    pub width: usize,
    pub height: usize,
    pub stride_x: usize,
    pub stride_y: usize,
}

impl ImageInfo {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            stride_x: 1,
            stride_y: width * 1,
        }
    }

    pub fn image_offset(&self, x: usize, y: usize) -> isize {
        (x * self.stride_x + y * self.stride_y) as isize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}
