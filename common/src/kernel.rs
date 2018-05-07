use super::{ImageInfo, Pixel};

pub fn blur(
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

    let mut r = 0.;
    let mut g = 0.;
    let mut b = 0.;
    let mut n = 0;
    for x2 in (x as isize - radius as isize)..(x as isize + radius as isize + 1) {
        for y2 in (y as isize - radius as isize)..(y as isize + radius as isize + 1) {
            if x2 < 0 || x2 >= info.width as isize || y2 < 0 || y2 >= info.height as isize {
                continue;
            }
            let p = unsafe { &*input_pixels.offset(info.image_offset(x2 as usize, y2 as usize)) };
            r += p.r as f64 / 0xFF as f64;
            g += p.g as f64 / 0xFF as f64;
            b += p.b as f64 / 0xFF as f64;
            n += 1;
        }
    }
    r /= n as f64;
    g /= n as f64;
    b /= n as f64;

    *output_pixel = Pixel {
        r: (r * 0xFF as f64) as u8,
        g: (g * 0xFF as f64) as u8,
        b: (b * 0xFF as f64) as u8,
    };
}
