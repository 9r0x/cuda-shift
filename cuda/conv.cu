#include "conv.h"

/* Convert HWC to CHW, then map each 0-255 value to 0.0-1.0 */
__host__ double *to_chw(unsigned char *input, int height, int width, int channels)
{
    double *output;
    catch_error(cudaMallocHost((void **)&output,
                               width * height * channels * sizeof(double)));
#pragma omp parallel for collapse(3)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                int oldIdx = TO_IDX_3(y, x, ch, width, channels);
                int newIdx = TO_IDX_3(ch, y, x, height, width);
                output[newIdx] = (double)(input[oldIdx]) / 255.0;
            }
        }
    }
    return output;
}

/* Map each 0.0-1.0 value to 0-255, then convert CHW to HWC */
__host__ unsigned char *to_hwc(double *input, int height, int width, int channels)
{
    unsigned char *output = new unsigned char[width * height * channels];
#pragma omp parallel for collapse(3)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                int oldIdx = TO_IDX_3(ch, y, x, height, width);
                int newIdx = TO_IDX_3(y, x, ch, width, channels);
                double value = input[oldIdx];
                value = fmin(value, 1.0);
                value = fmax(value, 0.0);
                output[newIdx] = (unsigned char)(fmin(fmax(value * 255.0,
                                                           0.0),
                                                      255.0) +
                                                 0.5);
            }
        }
    }
    return output;
}

/*
 Generate a 2D mask of size width x height, with 1.0 in the rectangle
 defined by top, bottom, left, right(boundary inclusive), and 0.0 elsewhere
 bottom = 0 means bottom = height - 1
 right = 0 means right = width - 1
 feather_distance is the no. of pixel the feathering effect,
 boundary exclusive(=0 means no feathering)
 only applied linearly outside the rectangle
*/
__global__ void gen_rectangle_mask(size_t height, size_t width,
                                   size_t top, size_t bottom,
                                   size_t left, size_t right,
                                   double *__restrict__ mask,
                                   size_t feather_distance)
{
    unsigned int y = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.z * blockDim.y + threadIdx.y;
    if (bottom == 0)
        bottom = height - 1;
    if (right == 0)
        right = width - 1;

    if ((x < width) && (y < height))
    {
        if ((x >= left) && (x <= right) && (y >= top) && (y <= bottom))
            mask[TO_IDX_2(y, x, width)] = 1.0;
        else
        {
            size_t edge_dist = feather_distance;
            if (x < left)
                edge_dist = left - x;
            else if (x > right)
                edge_dist = x - right;

            if (y < top)
                edge_dist = MIN(edge_dist, top - y);
            else if (y > bottom)
                edge_dist = MIN(edge_dist, y - bottom);

            mask[TO_IDX_2(y, x, width)] = edge_dist > feather_distance ? 0.0 : 1.0 - (double)edge_dist / (double)feather_distance;
        }
    }
}

__global__ void sum_masks(size_t height, size_t width, double *__restrict__ m1, double *__restrict__ m2, double *__restrict__ m3)
{
    unsigned int y = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.z * blockDim.y + threadIdx.y;

    if ((x < width) && (y < height))
    {
        double sum = m1[TO_IDX_2(y, x, width)] + m2[TO_IDX_2(y, x, width)] + m3[TO_IDX_2(y, x, width)];
        if (sum - 1.0 > 1e-6 || sum - 1.0 < -1e-6)
            printf("x: %d, y: %d: %f\n", x, y, sum);
    }
}

/*
 Performs a square 2D convolution with padding = 0, stride = 1
 Each image is array of channels x width x height
*/
__global__ void conv2D(double *__restrict__ input,
                       double *__restrict__ mask,
                       double *__restrict__ output,
                       size_t height, size_t width,
                       double *__restrict__ kernel, size_t kernel_radius)
{
    unsigned int c = blockIdx.x;
    // x and y are width and height of the input image
    unsigned int y = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.z * blockDim.y + threadIdx.y;
    size_t kernel_dim = (kernel_radius << 1) - 1;
    // Since we start from negative offset, use int
    ssize_t kernel_offset = kernel_radius - 1;

    if (x < width && y < height)
    {
        // Special case: simple pixel copy
        if (kernel_radius == 1 && kernel == NULL)
        {
            output[TO_IDX_3(c, y, x, height, width)] =
                input[TO_IDX_3(c, y, x, height, width)] * mask[TO_IDX_2(y, x, width)];
            return;
        }

        int output_idx = TO_IDX_3(c, y, x, height, width);
        if (mask[TO_IDX_2(y, x, width)] == 0.0)
        {
            output[output_idx] = 0.0;
            return;
        }

        double pixel_sum = 0.0;
        double normalization = 0.0;

        for (ssize_t ky = -kernel_offset; ky <= kernel_offset; ++ky)
        {
            ssize_t ny = (ssize_t)y + ky;
            if ((ny < 0) || (ny >= height))
                continue;

            for (ssize_t kx = -kernel_offset; kx <= kernel_offset; ++kx)
            {
                ssize_t nx = (ssize_t)x + kx;
                if ((nx < 0) || (nx >= width))
                    continue;

                size_t input_idx = TO_IDX_3(c, (size_t)ny, (size_t)nx,
                                            height, width);
                size_t kernel_idx = TO_IDX_2((size_t)(ky + kernel_offset),
                                             (size_t)(kx + kernel_offset),
                                             kernel_dim);
                size_t mask_idx = TO_IDX_2((size_t)ny, (size_t)nx, width);
                double coeff = kernel[kernel_idx] * mask[mask_idx];

                pixel_sum += coeff * input[input_idx];
                normalization += coeff;
            }
        }
        output[output_idx] = normalization ? pixel_sum / normalization * mask[TO_IDX_2(y, x, width)] : 0.0;
    }
}

/* Blend multiple images together */
__global__ void blend(double **__restrict__ input,
                      double *__restrict__ output,
                      size_t height, size_t width, size_t num)
{
    unsigned int c = blockIdx.x;
    unsigned int y = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.z * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        double pixel_sum = 0.0;
        size_t index = TO_IDX_3(c, y, x, height, width);
        for (size_t i = 0; i < num; ++i)
        {
            pixel_sum += input[i][index];
        }
        output[index] = pixel_sum;
    }
}