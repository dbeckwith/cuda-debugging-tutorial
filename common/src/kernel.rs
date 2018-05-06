use super::{ImageInfo, Pixel};

pub fn gaussian_blur(
    input_image: *const Pixel,
    output_image: *mut Pixel,
    info: *const ImageInfo,
    radius: f64,
    x: usize,
    y: usize,
) {
    let info = unsafe { &*info };

    if x >= info.width || y >= info.height {
        return;
    }

    let output_pixel = unsafe { &mut *output_image.offset(info.image_offset(x, y)) };

    *output_pixel = unsafe { *input_image.offset(info.image_offset(x, y)) };
}
