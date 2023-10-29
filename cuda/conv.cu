#include "conv.h"

/* Performs a square 2D convolution with padding = 0, stride = 1 */
__global__ void conv2D(float *input, float *output,
                       size_t width, size_t height, size_t channels,
                       double *kernel, size_t kernel_radius)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    double sum = 0.0;
}