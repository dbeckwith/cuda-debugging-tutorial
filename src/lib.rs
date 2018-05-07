#[macro_use]
extern crate error_chain;
extern crate common;
extern crate cuda;
extern crate image;

mod errors;
mod profiler;

use common::{ImageInfo, Pixel};
use cuda::driver as cu;
use errors::*;
use image::{ImageRgb8, RgbImage};
use profiler::Profiler;
use std::cmp::min;
use std::ffi::CString;
use std::mem::size_of;
use std::path::Path;

fn ceil_div(x: u32, y: u32) -> u32 {
    let mut n = x / y;
    if x % y != 0 {
        n += 1;
    }
    n
}

#[derive(Debug, Clone, Copy)]
struct WorkSize {
    x: u32,
    y: u32,
    z: u32,
}

fn partition_work(
    device: &cu::Device,
    kernel: &cu::Function,
    work: &WorkSize,
) -> Result<(cu::Grid, cu::Block)> {
    println!(
        "Work size: {}×{}×{} = {} items",
        work.x,
        work.y,
        work.z,
        work.x * work.y * work.z
    );

    let kernel_max_threads_per_block = kernel.max_threads_per_block()? as u32;
    let device_max_threads_per_block = device.max_threads_per_block()? as u32;
    let max_block = cu::Block::xyz(
        device.max_block_dim_x()? as u32,
        device.max_block_dim_y()? as u32,
        device.max_block_dim_z()? as u32,
    );
    let max_threads_per_block = min(device_max_threads_per_block, kernel_max_threads_per_block);
    let max_registers_per_block = device.max_registers_per_block()? as u32;
    let registers_per_thread = kernel.num_regs()? as u32;

    println!(
        "Kernel max threads per block: {} threads",
        kernel_max_threads_per_block
    );
    println!(
        "Device max threads per block: {} threads",
        device_max_threads_per_block
    );
    println!(
        "Device max block size: {}x{}x{} threads",
        max_block.x, max_block.y, max_block.z
    );
    println!(
        "Device max registers per block: {} registers",
        max_registers_per_block
    );
    println!("Registers per thread: {} registers", registers_per_thread);

    // TODO: consider that this may not be optimal
    // first of all, it doesn't even maximize threads_per_block
    // second, maximal threads_per_block isn't necessarily optimal
    // block size should be based on how the kernel will branch
    // e.g. if the kernel always branches the same on all z values,
    // but randomly on x and y values,
    // then optimal block size would be 1×1×∞

    let check = |block: &cu::Block| {
        if block.x > work.x {
            return false;
        }
        if block.x > max_block.x {
            return false;
        }
        if block.y > work.y {
            return false;
        }
        if block.y > max_block.y {
            return false;
        }
        if block.z > work.z {
            return false;
        }
        if block.z > max_block.z {
            return false;
        }
        let threads_per_block = block.x * block.y * block.z;
        if threads_per_block > max_threads_per_block {
            return false;
        }
        let registers_per_block = registers_per_thread * threads_per_block;
        if registers_per_block > max_registers_per_block {
            return false;
        }
        true
    };

    let mut block = cu::Block::xyz(1, 1, 1);
    loop {
        let mut failed_checks = 0;
        block.x += 1;
        if !check(&block) {
            block.x -= 1;
            failed_checks += 1;
        }
        block.y += 1;
        if !check(&block) {
            block.y -= 1;
            failed_checks += 1;
        }
        block.z += 1;
        if !check(&block) {
            block.z -= 1;
            failed_checks += 1;
        }
        if failed_checks == 3 {
            break;
        }
    }

    println!(
        "Block size: {}×{}×{} = {} threads",
        block.x,
        block.y,
        block.z,
        block.x * block.y * block.z
    );

    let grid = cu::Grid::xyz(
        ceil_div(work.x, block.x),
        ceil_div(work.y, block.y),
        ceil_div(work.z, block.z),
    );

    println!(
        "Grid size: {}×{}×{} = {} blocks",
        grid.x,
        grid.y,
        grid.z,
        grid.x * grid.y * grid.z
    );

    let threads_per_block = block.x * block.y * block.z;
    let registers_per_block = registers_per_thread * threads_per_block;

    println!(
        "Threads per block: {} threads ({:.2}%)",
        threads_per_block,
        threads_per_block as f32 / max_threads_per_block as f32 * 100.
    );
    println!(
        "Registers per block: {} registers ({:.2}%)",
        registers_per_block,
        registers_per_block as f32 / max_registers_per_block as f32 * 100.
    );

    assert!(threads_per_block <= max_threads_per_block);
    assert!(registers_per_block <= max_registers_per_block);

    Ok((grid, block))
}

pub fn run_blur(radius: f64, input_path: &Path, output_path: &Path) -> Result<()> {
    let mut profiler = Profiler::new();

    profiler.step("Reading input image");
    let input_image = image::open(input_path).chain_err(|| "Could not open image")?;
    let input_image = match input_image {
        ImageRgb8(img) => img,
        _ => {
            profiler.step("Converting input image to RGB");
            input_image.to_rgb()
        }
    };

    let image_info = ImageInfo::new(input_image.width() as usize, input_image.height() as usize);

    println!("Image size: {}×{}", image_info.width, image_info.height);

    profiler.step("Initializing CUDA driver");
    cu::initialize().chain_err(|| "Error initializing CUDA driver")?;

    let device_count = cu::Device::count()?;
    println!("");
    println!("Total devices: {}", device_count);
    println!("");

    assert!(device_count > 0);

    for device_idx in 0..device_count {
        let device = cu::Device(device_idx as u16)?;

        println!(
            "Device {}: {}",
            device_idx,
            device.name()?.to_string_lossy().into_owned()
        );
        if device.multi_gpu_board()? {
            println!("Multi-GPU group ID: {}", device.multi_gpu_board_group_id()?);
        }
        println!(
            "Total memory: {} MiB",
            device.total_memory()? as f32 / (1 << 20) as f32
        );
        println!(
            "Memory clock rate: {} MHz",
            device.memory_clock_rate()? as f32 / 1000.
        );
        println!("Multiprocessors: {}", device.multiprocessor_count()?);
        println!("Clock rate: {} MHz", device.clock_rate()? as f32 / 1000.);
        println!("");
    }

    profiler.step("Initializing CUDA context");
    let device = cu::Device(0).chain_err(|| "Error initializing CUDA device 0")?;
    let context = device
        .create_context()
        .chain_err(|| "Error initializing device context")?;

    profiler.step("Loading kernels");
    println!("PTX source from {}", env!("KERNEL_PTX_PATH"));
    let ptx_bytecode = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).unwrap();
    let module = context
        .load_module(&ptx_bytecode)
        .chain_err(|| "Error loading PTX module")?;
    let blur_kernel = module.function(&CString::new("gaussian_blur").unwrap())?;

    context.set_current()?;

    profiler.step("Initializing device buffers");
    let image_info_buf = unsafe {
        cu::allocate(size_of::<ImageInfo>())
            .chain_err(|| "Error allocating device buffer for image info")?
            as *mut ImageInfo
    };
    let input_pixels_buf = unsafe {
        cu::allocate(input_image.len() * size_of::<Pixel>())
            .chain_err(|| "Error allocating device buffer for input pixel data")?
            as *mut Pixel
    };
    let output_pixels_buf = unsafe {
        cu::allocate(input_image.len() * size_of::<Pixel>())
            .chain_err(|| "Error allocating device buffer for output pixel data")?
            as *mut Pixel
    };

    profiler.step("Copying data to device");
    unsafe {
        cu::copy(
            &image_info as *const ImageInfo,
            image_info_buf,
            1,
            cu::Direction::HostToDevice,
        ).chain_err(|| "Error copying image info to device")?;

        cu::copy(
            input_image.as_ptr() as *const Pixel,
            input_pixels_buf,
            input_image.len() * size_of::<u8>() / size_of::<Pixel>(),
            cu::Direction::HostToDevice,
        ).chain_err(|| "Error copying input image to device")?;
    }

    profiler.step("Partitioning work");
    let (grid, block) = partition_work(
        &device,
        &blur_kernel,
        &WorkSize {
            x: image_info.width as u32,
            y: image_info.height as u32,
            z: 1,
        },
    )?;

    profiler.step("Running blur kernel");
    blur_kernel
        .launch(
            &[
                cu::Any(&input_pixels_buf),
                cu::Any(&output_pixels_buf),
                cu::Any(&image_info_buf),
                cu::Any(&radius),
            ],
            grid,
            block,
        )
        .chain_err(|| "Error running render kernel")?;

    profiler.step("Copying data from device");
    let mut output_image = RgbImage::new(image_info.width as u32, image_info.height as u32);
    unsafe {
        cu::copy(
            output_pixels_buf,
            output_image.as_mut_ptr() as *mut Pixel,
            output_image.len() * size_of::<u8>() / size_of::<Pixel>(),
            cu::Direction::DeviceToHost,
        ).chain_err(|| "Error copying output image from device")?;
    }

    profiler.step("Saving image");
    ImageRgb8(output_image)
        .save(output_path)
        .chain_err(|| "Error saving output image")?;

    Ok(())
}
