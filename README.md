# Tilt Shift Effect with Parallel Computing

Shukai Ni, Jiayu Zheng

# Examples

## Manual

**Original Image**
![Example](./manual/original.jpg)

**Tilt Shift Effect with GIMP**
![Example](./manual/tiltshift.jpg)

## With CUDA

**TODO**

# Usage

```
./tiltshift <input> <output> <radius> <sigma>
```

# Gaussian kernel

## Kernel generation

- $r$ is the radius of the kernel(including the center pixel)
- sigma is the standard deviation of the Gaussian function

We could use 1D Gaussian kernel to generate 2D Gaussian kernel:

$$ g(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{x^2}{2\sigma^2}} $$

The 2D Gaussian kernel is the outer product of two 1D Gaussian kernels.

After generating half of the 1D Gaussian kernel, we compute one eighth of the 2D Gaussian kernel. Then taking advantage of the symmetry, we can mirror the rest of the 2D Gaussian kernel.

In theory, we could mirror 1/8, then 1/4, then 1/2 of the entire kernel to take advantage of sequential memory access. However, to simplify the implementation and assuming a reasonably large kernel size, we will only mirror the entire kernel in parallel with OpenMP.

## Normalization

The overall brightness of an image should remain consistent after applying a Gaussian blur if the Gaussian kernel is normalized. A normalized Gaussian kernel has elements that sum up to 1. This ensures that when it's applied to an image, the total sum of pixel values in the filtered image remains approximately the same as in the original image.

However, due to the discrete nature of images and the approximation in the Gaussian blur, slight changes in brightness can still occur. This is because the Gaussian kernel is only an approximation of the Gaussian function, and the kernel is only applied to a finite number of pixels in the image.

Like before, we could use OpenMP to normalize the Gaussian kernel in parallel. In particular, we could use the reduction clause to compute the sum of all elements in the kernel.

# TODO

- [x] Manual effect with GIMP
- [ ] Python effect
- [ ] Gaussian kernel CUDA
- [ ] Gaussian dot product CUDA
- [ ] Gaussian warp shared memory CUDA
