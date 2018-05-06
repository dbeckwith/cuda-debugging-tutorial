error_chain!{
    foreign_links {
        Cuda(::cuda::driver::Error);
        Image(::image::ImageError);
        Io(::std::io::Error);
    }
}
