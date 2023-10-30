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

    size_t in_width, in_height, out_width, out_height;
    size_t channels, kernel_radius, kernel_dim;
    // No. of total bytes on the device
    size_t kernel_bytes, in_bytes, out_bytes;
    const char *input_path, *output_path;
    unsigned char *image;
    double sigma;
    double *kernel_h, *in_h, *out_h;
    double *kernel_d, *in_d, *out_d;

    double *near_mask_d, *mid_mask_d, *far_mask_d;
    double *near_out_d, *mid_out_d, *far_out_d;

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

    image = load_image(input_path, &in_width, &in_height, &channels);
    if (__glibc_unlikely(!image))
    {
        fprintf(stderr, "[!] Error loading image.\n");
        return 1;
    }
    printf("[+] Image loaded from %s\n", input_path);

    if (__glibc_unlikely(in_width < kernel_dim || in_height < kernel_dim))
    {
        fprintf(stderr, "[!] Width or height is smaller than kernel dimension.\n");
        return 1;
    }

    in_bytes = in_width * in_height * channels * sizeof(double);
    // apply gaussian will lose the padding
    out_width = in_width - kernel_dim + 1;
    out_height = in_height - kernel_dim + 1;
    out_bytes = out_width * out_height * channels * sizeof(double);

    printf("[+] Kernel dimensions: %d x %d \n", kernel_dim, kernel_dim);
    printf("[+] Input dimensions: %d x %d x %d\n", in_width, in_height, channels);
    printf("[+] Output dimensions: %d x %d x %d\n", out_width, out_height, channels);

    kernel_h = gen_gaussian_kernel(kernel_radius, sigma);
    mirror_gaussian_kernel(kernel_radius, kernel_h);
    normalize_gaussian_kernel(kernel_radius, kernel_h);

    // Convert whc to cwh
    in_h = to_cwh(image, in_width, in_height, channels);
    if (__glibc_unlikely(!in_h))
    {
        free_image(image);
        fprintf(stderr, "[!] Error converting image to cwh format.\n");
        return 1;
    }
    free_image(image);

    // Prepare GPU
    dim3 blockSize(32, 32);
    dim3 gridSize(channels,
                  (in_height + blockSize.x - 1) / blockSize.x,
                  (in_width + blockSize.y - 1) / blockSize.y);

    printf("[+] Grid size: %d, %d, %d\n", gridSize.x, gridSize.y, gridSize.z);
    printf("[+] Block size: %d, %d\n", blockSize.x, blockSize.y);

    catch_error(cudaMalloc((void **)&kernel_d, kernel_bytes));
    catch_error(cudaMalloc((void **)&in_d, in_bytes));
    catch_error(cudaMalloc((void **)&out_d, out_bytes));

    catch_error(cudaMemcpy(kernel_d, kernel_h, kernel_bytes, cudaMemcpyHostToDevice));
    catch_error(cudaMemcpy(in_d, in_h, in_bytes, cudaMemcpyHostToDevice));

    // Generate near, mid, far masks
    catch_error(cudaMalloc((void **)&near_mask_d, in_width * in_height * sizeof(double)));
    catch_error(cudaMalloc((void **)&mid_mask_d, in_width * in_height * sizeof(double)));
    catch_error(cudaMalloc((void **)&far_mask_d, in_width * in_height * sizeof(double)));

    catch_error(cudaMalloc((void **)&near_out_d, in_bytes));
    catch_error(cudaMalloc((void **)&mid_out_d, in_bytes));
    catch_error(cudaMalloc((void **)&far_out_d, in_bytes));

    printf("[+] Memory allocated on device.\n");

    // FIX ME
    gen_rectangle_mask<<<gridSize, blockSize>>>(in_width, in_height,
                                                0, 7978, 0, 2000,
                                                1, near_mask_d);
    gen_rectangle_mask<<<gridSize, blockSize>>>(in_width, in_height,
                                                0, 7978, 2001, 4000,
                                                1, mid_mask_d);
    gen_rectangle_mask<<<gridSize, blockSize>>>(in_width, in_height,
                                                0, 7978, 4001, 6000,
                                                1, far_mask_d);
    catch_error(cudaDeviceSynchronize());
    printf("[+] Masks generated.\n");

    apply_mask<<<gridSize, blockSize>>>(in_d, near_mask_d, near_out_d,
                                        in_width, in_height, channels);
    apply_mask<<<gridSize, blockSize>>>(in_d, mid_mask_d, mid_out_d,
                                        in_width, in_height, channels);
    apply_mask<<<gridSize, blockSize>>>(in_d, far_mask_d, far_out_d,
                                        in_width, in_height, channels);
    catch_error(cudaDeviceSynchronize());
    printf("[+] Masks applied.\n");

    // Apply gaussian
    conv2D<<<gridSize, blockSize>>>(in_d, out_d,
                                    in_width, in_height, channels,
                                    kernel_d, kernel_radius);

    catch_error(cudaMallocHost((void **)&out_h, out_bytes));
    catch_error(cudaDeviceSynchronize());
    catch_error(cudaMemcpy(out_h, out_d, out_bytes, cudaMemcpyDeviceToHost));
    printf("[+] Convolution done.\n");

#ifdef DEBUG
    for (auto i = 0; i < kernel_dim; i++)
    {
        for (auto j = 0; j < kernel_dim; j++)
            printf("%f ", kernel_h[i * kernel_dim + j]);
        printf("\n");
    }
#endif

    // Reusing freed image pointer
    image = to_whc(out_h, out_width, out_height, channels);
    // stb returns 0 on failure
    if (__glibc_unlikely(!save_image(output_path, image, out_width, out_height, channels)))
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

    catch_error(cudaFree(kernel_d));
    catch_error(cudaFree(in_d));

    catch_error(cudaFreeHost(out_h));
    catch_error(cudaFree(out_d));
    return 0;
}
