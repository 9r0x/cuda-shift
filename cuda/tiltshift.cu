#include <cuda.h>
#include <stdio.h>

#include "kernel.h"
#include "image.h"
#include "conv.h"
#include "utils.h"

#define FULL_MASK 0xffffffff
#define __MAXWARPS_PER_BLOCK__ (__MAXLANES_PER_BLOCK__ / __LANES_PER_WARP__)

#define BLOCK_DIM 32
#define __MAXLANES_PER_BLOCK__ (BLOCK_DIM * BLOCK_DIM)
#define __LANES_PER_WARP__ 32 // Nvidia defines this as 32

__host__ void printUsage(const char *programName)
{
    fprintf(stderr, "Usage: %s <image_path> <kernel_dim> <sigma> <output_image>\n", programName);
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  <input_path>    Path to the input image. Supported formats are PNG and JPEG(JPG).\n");
    fprintf(stderr, "  <kernel_radius> Dimension of the Gaussian kernel radius, center pixel included.\n");
    fprintf(stderr, "  <sigma>         Standard deviation of the Gaussian kernel. Must be a positive number.\n");
    fprintf(stderr, "  <output_path>   Path to the output image. Supported formats are PNG and JPEG(JPG).\n");
    fprintf(stderr, "\nExample: %s input.png 5 1.0 output.png\n", programName);
}

int main(int argc, char **argv)
{

    size_t width, height;
    size_t channels, kernel_radius, kernel_dim;
    // No. of total bytes on the device
    size_t kernel_bytes, image_bytes, mask_bytes;
    const char *input_path, *output_path;
    unsigned char *image;
    double sigma;
    double *kernel_h, *in_h, *out_h;
    double *kernel_d, *in_d, *out_d;

    double *near_mask_d, *mid_mask_d, *far_mask_d;
    double *near_out_d, *mid_out_d, *far_out_d;

    // Although this is a on-host array
    // its elements are pointers to device memory
    double *in_blend_h[3];
    double **in_blend_d;

    if (__glibc_unlikely(argc != 5))
    {
        printUsage(argv[0]);
        return 1;
    }

    input_path = argv[1];
    kernel_radius = atoi(argv[2]);
    sigma = atof(argv[3]);
    output_path = argv[4];

    if (__glibc_unlikely(kernel_radius < 1))
    {
        fprintf(stderr, "[!] Kernel radius must be a positive integer.\n");
        return 1;
    }
    kernel_dim = (kernel_radius << 1) - 1;
    kernel_bytes = kernel_dim * kernel_dim * sizeof(double);

    if (__glibc_unlikely(sigma <= 0))
    {
        fprintf(stderr, "[!] Sigma must be a positive number.\n");
        return 1;
    }

    printf("\n[ ] Loading image ...\n");
    image = load_image(input_path, &height, &width, &channels);
    if (__glibc_unlikely(!image))
    {
        fprintf(stderr, "[!] Error loading image.\n");
        return 1;
    }
    printf("[+] Image loaded from %s\n", input_path);

    mask_bytes = width * height * sizeof(double);
    image_bytes = mask_bytes * channels;

    printf("[*] Kernel dimensions: %d x %d \n", kernel_dim, kernel_dim);
    printf("[*] Image dimensions(HWC): %d x %d x %d\n", height, width, channels);

    printf("\n[ ] Generating Gaussian kernel ...\n");
    kernel_h = gen_gaussian_kernel(kernel_radius, sigma);
    mirror_gaussian_kernel(kernel_radius, kernel_h);
    normalize_gaussian_kernel(kernel_radius, kernel_h);

#ifdef DEBUG
    for (auto i = 0; i < kernel_dim; i++)
    {
        for (auto j = 0; j < kernel_dim; j++)
            printf("%f ", kernel_h[i * kernel_dim + j]);
        printf("\n");
    }
#endif
    printf("[+] Kernel generated.\n");

    printf("\n[ ] Converting image to cwh format ...\n");
    in_h = to_chw(image, height, width, channels);
    if (__glibc_unlikely(!in_h))
    {
        free_image(image);
        fprintf(stderr, "[!] Error converting image to cwh format.\n");
        return 1;
    }
    free_image(image);
    printf("[+] Image converted.\n");

    printf("\n[ ] Allocating memory on device ...\n");
    dim3 blockSize(32, 32);
    dim3 gridSize(channels,
                  (height + blockSize.x - 1) / blockSize.x,
                  (width + blockSize.y - 1) / blockSize.y);

    printf("[*] Grid size: %d, %d, %d\n", gridSize.x, gridSize.y, gridSize.z);
    printf("[*] Block size: %d, %d\n", blockSize.x, blockSize.y);

    catch_error(cudaMallocHost((void **)&out_h, image_bytes));

    catch_error(cudaMalloc((void **)&kernel_d, kernel_bytes));
    catch_error(cudaMalloc((void **)&in_d, image_bytes));
    catch_error(cudaMalloc((void **)&out_d, image_bytes));

    catch_error(cudaMemcpy(kernel_d, kernel_h, kernel_bytes, cudaMemcpyHostToDevice));
    catch_error(cudaMemcpy(in_d, in_h, image_bytes, cudaMemcpyHostToDevice));

    catch_error(cudaMalloc((void **)&near_mask_d, mask_bytes));
    catch_error(cudaMalloc((void **)&mid_mask_d, mask_bytes));
    catch_error(cudaMalloc((void **)&far_mask_d, mask_bytes));

    catch_error(cudaMalloc((void **)&near_out_d, image_bytes));
    catch_error(cudaMalloc((void **)&mid_out_d, image_bytes));
    catch_error(cudaMalloc((void **)&far_out_d, image_bytes));
    printf("[+] Memory allocated on device.\n");

    printf("\n[ ] Generating masks ...\n");
    // TODO dynamically adjust and use Macro/var instead of magic number
    size_t feather_radius = kernel_radius * 4;
    gen_rectangle_mask<<<gridSize, blockSize>>>(height, width,
                                                0, height / 5, 0, 0,
                                                near_mask_d, feather_radius);
    gen_rectangle_mask<<<gridSize, blockSize>>>(height, width,
                                                height / 5 + feather_radius, 4 * height / 5, 0, 0,
                                                mid_mask_d, feather_radius);
    gen_rectangle_mask<<<gridSize, blockSize>>>(height, width,
                                                4 * height / 5 + feather_radius, 0, 0, 0,
                                                far_mask_d, feather_radius);
    catch_error(cudaDeviceSynchronize());
    printf("[+] Masks generated.\n");

    // sum masks and all should be 1
    sum_masks<<<gridSize, blockSize>>>(height, width,
                                       near_mask_d, mid_mask_d, far_mask_d);

    printf("\n[ ] Applying convolution ...\n");
    conv2D<<<gridSize, blockSize>>>(in_d, near_mask_d, near_out_d,
                                    height, width,
                                    kernel_d, kernel_radius);

    conv2D<<<gridSize, blockSize>>>(in_d, mid_mask_d, mid_out_d,
                                    height, width,
                                    NULL, 1);

    conv2D<<<gridSize, blockSize>>>(in_d, far_mask_d, far_out_d,
                                    height, width,
                                    kernel_d, kernel_radius);
    catch_error(cudaDeviceSynchronize());
    printf("[+] Convolution applied.\n");

    printf("\n[ ] Combining images ...\n");
    in_blend_h[0] = near_out_d;
    in_blend_h[1] = mid_out_d;
    in_blend_h[2] = far_out_d;

    catch_error(cudaMalloc((void **)&in_blend_d, 3 * sizeof(double *)));
    catch_error(cudaMemcpy(in_blend_d, in_blend_h,
                           3 * sizeof(double *), cudaMemcpyHostToDevice));

    blend<<<gridSize, blockSize>>>(in_blend_d, out_d, height, width, 3);

    catch_error(cudaDeviceSynchronize());
    catch_error(cudaMemcpy(out_h, out_d, image_bytes, cudaMemcpyDeviceToHost));
    printf("[+] Images combined.\n");

    printf("\n[ ] Saving image ...\n");
    // Reusing freed image pointer
    image = to_hwc(out_h, height, width, channels);
    // stb returns 0 on failure
    if (__glibc_unlikely(!save_image(output_path, image, height, width, channels)))
    {
        free_image(image);
        fprintf(stderr, "[!] Error saving image.\n");
        return 1;
    }
    else
    {
        free_image(image);
        printf("[+] Image saved to %s\n", output_path);
    }

    catch_error(cudaFreeHost(kernel_h));
    catch_error(cudaFreeHost(in_h));
    catch_error(cudaFreeHost(out_h));

    catch_error(cudaFree(kernel_d));
    catch_error(cudaFree(in_blend_d));
    catch_error(cudaFree(near_mask_d));
    catch_error(cudaFree(mid_mask_d));
    catch_error(cudaFree(far_mask_d));
    catch_error(cudaFree(near_out_d));
    catch_error(cudaFree(mid_out_d));
    catch_error(cudaFree(far_out_d));
    catch_error(cudaFree(in_d));
    catch_error(cudaFree(out_d));
    return 0;
}
