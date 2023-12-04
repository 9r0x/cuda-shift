# Tilt Shift Effect with Parallel Computing

# Examples

**Original Image**
![Example](./Figures/Original.jpg)

**Tilt Shift Effect with GIMP**
![Example](./Figures/Tiltshift.jpg)

# Build

```shell
cd cuda
make -j`nproc`
# debug build
# debug=1 make -j`nproc`
```

# Usage

```
Usage: ./cuda/bin/tiltshift <input_path> <output_path>
Arguments:
  <input_path>    Path to the input image. Supported formats are PNG and JPEG(JPG).
  <output_path>   Path to the output image. Supported formats are PNG and JPEG(JPG).
Options:
  --help         -h         Print this help message.
  --radius       -r = 10    Dimension of the Gaussian kernel radius, center pixel included.
  --sigma        -s = 3.0   Standard deviation of the Gaussian kernel. Must be a positive number.
  --far          -f = 0.2   Lower end of the far mask in height ratio.
  --near         -n = 0.8   Upper end of the near mask in height ratio.
  --feather         = 0.1   Feather size in height ratio.

Example: ./cuda/bin/tiltshift input.png -r 5 - s 1.2 output.png
```

# Global Workflow

1. Extract Near, Mid, Far region masks from the input image
2. Apply Gaussian blur to each mask
3. Blend the blurred masks together to get the final image

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

## Normalization for advanced masks

In the goal state, user should be able to pass in a mask to specify the region of interest. In this case, the normalization should be done differently. Instead of normalizing the entire kernel, we should only normalize the kernel during the application. This way the kernel can be normalized based on the sum of brightness of the pixels in the region of interest.
K is the kernel matrix, M is the neighboring mask, N reprents neighboring pixels.

$$p_{ij} = \sum K_{ij} \cdot M_{ij} \cdot N_{ij}$$

For center pixel $p$ have similar brightness as the original image, we need to perform coefficient normalization:

$$K_{ij} = \frac{\sum K_{ij} \cdot M_{ij} \cdot N_{ij}}{\sum K_{ij} \cdot M_{ij}}$$

This solution avoids the problem of padding at the edge of the image because the edge will have one-sided pixels. It also adds support for arbitrary mask shapes and values.

# TODO

- [x] Manual effect with GIMP
- [x] Python effect
- [x] kernel generation with OpenMP
- [x] 2D Convolution in CUDA
- [x] Rectangle mask generation with OpenMP
- [x] Mask application in CUDA
- [x] Per mask Gaussian blur in CUDA
- [x] Mask blending in CUDA
- [x] Refactored CUDA functions
- [x] Generate masks that is semi-transparent at the edge

- [x] Improve per pixel dot product in CUDA

# References

- [stb_image](https://github.com/nothings/stb/) for image loading and saving
