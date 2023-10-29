#include <cuda.h>
#include <stdio.h>

#include "gaussian.h"
#include "image.h"

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

__host__ void catch_error(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[!] CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv)
{

    size_t width, height, channels, kernel_radius;
    double sigma;
    const char *input_path, *output_path;
    unsigned char *image;
    double *gaussian_kernel_h, *image_h;

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

    if (__glibc_unlikely(sigma <= 0))
    {
        fprintf(stderr, "[!] Sigma must be a positive number.\n");
        return 1;
    }

    image = load_image(input_path, &width, &height, &channels);
    if (__glibc_unlikely(!image))
    {
        fprintf(stderr, "[!] Error loading image.\n");
        return 1;
    }

    printf("[+] Image loaded from %s\n", input_path);
    printf("[+] Image dimensions: %d x %d x %d\n", width, height, channels);

    gaussian_kernel_h = gen_gaussian_kernel(kernel_radius, sigma);
    mirror_gaussian_kernel(kernel_radius, gaussian_kernel_h);
    normalize_gaussian_kernel(kernel_radius, gaussian_kernel_h);

    image_h = to_cwh(image, width, height, channels);
    if (__glibc_unlikely(!image_h))
    {
        free_image(image);
        fprintf(stderr, "[!] Error converting image to cwh format.\n");
        return 1;
    }
    free_image(image);

    // Begin Applying the Gaussian Filter
    dim3 blockSize(32, 32, 1);
    dim3 gridSize(channels, (width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    printf("[+] Grid size: %d, %d\n", gridSize.x, gridSize.y);
    printf("[+] Block size: %d, %d\n", blockSize.x, blockSize.y);

    /* TODO
    dim3 gridSize((n + BLOCK_DIM - 1) / BLOCK_DIM,
                  (n + BLOCK_DIM - 1) / BLOCK_DIM);
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
#ifdef DEBUG

#endif

catch_error(cudaFree(gaussian_kernel_d));
*/

#ifdef DEBUG
    size_t kernel_dim = (kernel_radius << 1) - 1;
    for (auto i = 0; i < kernel_dim; i++)
    {
        for (auto j = 0; j < kernel_dim; j++)
            printf("%f ", gaussian_kernel_h[i * kernel_dim + j]);
        printf("\n");
    }
#endif

    // Reusing freed image pointer
    image = to_whc(image_h, width, height, channels);
    // stb returns 0 on failure
    if (__glibc_unlikely(!save_image(output_path, image, width, height, channels)))
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

    delete[] gaussian_kernel_h;
    return 0;
}
