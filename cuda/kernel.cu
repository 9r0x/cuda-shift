#include "kernel.h"

/* Generate upper half of the lower right quarter of the kernel(1/8 of the kernel) */
__host__ double *gen_gaussian_kernel(size_t r, double sigma)
{
    size_t kernel_dim = ((r << 1) - 1);
    size_t kernel_size = kernel_dim * kernel_dim;

    double *gaussian_1d = new double[r];
    double *gaussian_2d;
    double two_sigma_squared = sigma * sigma * 2;

    catch_error(cudaMallocHost((void **)&gaussian_2d, kernel_size * sizeof(double)));

    if (__glibc_unlikely(!gaussian_1d || !gaussian_2d))
    {
        fprintf(stderr, "Failed to allocate memory for gaussian kernel array(s)\n");
        exit(1);
    }

#define sqrt_2pi 2.506628274631000241612355239340104162693023681640625
#pragma omp parallel for
    for (auto i = 0; i < r; i++)
        gaussian_1d[i] = exp(-((double)(i * i)) / (two_sigma_squared)) / sqrt_2pi / sigma;
#undef sqrt_2pi

    // (2*r-1) * (r-1) + (r-1) === (r-1) * (2*r)
    // shift is faster than multiplication
    size_t base_idx = ((r - 1) * r) << 1;
#pragma omp parallel for
    for (auto i = 0; i < r; i++)
    {
        double gi = gaussian_1d[i];
        for (auto j = i; j < r; j++)
        {
            // Computer upper half of the lower right quarter of the kernel(1/8 of the kernel)
            double g = gaussian_1d[j] * gi;
            gaussian_2d[base_idx + j] = g;
        }
        //  add within each iter to reduce computation
        base_idx += kernel_dim;
    }

    delete[] gaussian_1d;
    return gaussian_2d;
}

/* Mirror 1/8 to the entire kernel in parallel */
__host__ void mirror_gaussian_kernel(size_t r, double *gaussian_kernel_h)
{
    size_t kernel_dim = (r << 1) - 1;

#pragma omp parallel for collapse(2)
    for (auto i = 0; i < kernel_dim; ++i)
    {
        for (auto j = 0; j < kernel_dim; ++j)
        {
            int i_mirror = i < r - 1 ? kernel_dim - i - 1 : i;
            int j_mirror = j < r - 1 ? kernel_dim - j - 1 : j;

            // If we are at the lower left half of the lower right quarter
            // mirror over diagonal(i.e. swap)
            if (i_mirror > j_mirror)
            {
                int tmp = i_mirror;
                i_mirror = j_mirror;
                j_mirror = tmp;
            }
            // If not using OpenMP, take constant out of the inner loop
            gaussian_kernel_h[TO_IDX_2(i, j, kernel_dim)] =
                gaussian_kernel_h[TO_IDX_2(i_mirror, j_mirror, kernel_dim)];
        }
    }
}

/* Normalize the kernel so that the sum of all elements is 1 */
__host__ void normalize_gaussian_kernel(size_t r, double *gaussian_kernel_h)
{
    size_t kernel_dim = (r << 1) - 1;
    size_t kernel_size = kernel_dim * kernel_dim;

    double sum = 0;

#pragma omp parallel for reduction(+ : sum)
    for (auto i = 0; i < kernel_size; i++)
        sum += gaussian_kernel_h[i];

#pragma omp parallel for
    for (auto i = 0; i < kernel_size; i++)
        gaussian_kernel_h[i] /= sum;
}
