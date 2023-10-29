#include <cuda.h>
#include <stdio.h>

#include "gaussian.h"

#define FULL_MASK 0xffffffff
#define __MAXWARPS_PER_BLOCK__ (__MAXLANES_PER_BLOCK__ / __LANES_PER_WARP__)

#define BLOCK_DIM 32
#define __MAXLANES_PER_BLOCK__ (BLOCK_DIM * BLOCK_DIM)
#define __LANES_PER_WARP__ 32 // Nvidia defines this as 32

__host__ void catch_error(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (__glibc_unlikely(argc != 3))
    {
        // TODO
        // fprintf(stderr, "Usage: %s <image_width> <image_height> <kernel_dim> <sigma> <output_image>\n", argv[0]);
        fprintf(stderr, "Usage: %s <kernel_radius> <sigma> \n", argv[0]);
        return 1;
    }

    size_t kernel_radius = atoi(argv[1]);
    double sigma = atof(argv[2]);
    double *gaussian_kernel_h = gen_gaussian_kernel(kernel_radius, sigma);
    mirror_gaussian_kernel(kernel_radius, gaussian_kernel_h);
    normalize_gaussian_kernel(kernel_radius, gaussian_kernel_h);

    /* TODO
        dim3 gridSize((n + BLOCK_DIM - 1) / BLOCK_DIM,
                      (n + BLOCK_DIM - 1) / BLOCK_DIM);
        dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    #ifdef DEBUG
        printf("gridSize: %d, %d\n", gridSize.x, gridSize.y);
        printf("blockSize: %d, %d\n", blockSize.x, blockSize.y);
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

    // TODO apply gaussian kernel to image
    delete[] gaussian_kernel_h;

    return 0;
}
