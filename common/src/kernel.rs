use super::{ImageInfo, Pixel};

pub fn gaussian_blur(
    input_pixels: *const Pixel,
    output_pixels: *mut Pixel,
    info: *const ImageInfo,
    radius: f64,
    x: usize,
    y: usize,
) {
    let info = unsafe { &*info };

    if x >= info.width || y >= info.height {
        return;
    }

    let output_pixel = unsafe { &mut *output_pixels.offset(info.image_offset(x, y)) };

    *output_pixel = unsafe { *input_pixels.offset(info.image_offset(x, y)) };
}
