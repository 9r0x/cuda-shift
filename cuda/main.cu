#include <cuda.h>
#include <stdio.h>

#define FULL_MASK 0xffffffff
#define __MAXLANES_PER_BLOCK__ 1024
#define __LANES_PER_WARP__ 32
#define __MAXWARPS_PER_BLOCK__ (__MAXLANES_PER_BLOCK__ / __LANES_PER_WARP__)
#define THREADX 256 // Need to be less than __MAXLANES_PER_BLOCK__

#define N 10000

__global__ void dot_kernel(double *x_d, double *result_d)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    double sum = 0;

    if (i < N)
        sum = x_d[i] * x_d[i];
    __syncwarp();

    for (int offset = __LANES_PER_WARP__ / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(FULL_MASK, sum, offset);

    __shared__ float s_mem[__MAXWARPS_PER_BLOCK__];
    int nwarps = blockDim.x / __LANES_PER_WARP__;
    int warp_ID = threadIdx.x / __LANES_PER_WARP__;
    int laneID_in_warp = threadIdx.x % __LANES_PER_WARP__;
    if (laneID_in_warp == 0)
        s_mem[warp_ID] = sum;

    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int i = 1; i < nwarps; ++i)
            sum += s_mem[i];

        result_d[blockIdx.x] = sum;
    }
}

__global__ void sum_kernel(double *result_d, double *sum_d)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    double sum = 0;

    if (i < blockDim.x)
        sum = result_d[i];
    __syncwarp();

    for (int offset = __LANES_PER_WARP__ / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(FULL_MASK, sum, offset);

    __shared__ float s_mem[__MAXWARPS_PER_BLOCK__];
    int nwarps = blockDim.x / __LANES_PER_WARP__;
    int warp_ID = threadIdx.x / __LANES_PER_WARP__;
    int laneID_in_warp = threadIdx.x % __LANES_PER_WARP__;
    if (laneID_in_warp == 0)
        s_mem[warp_ID] = sum;

    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int i = 1; i < nwarps; ++i)
            sum += s_mem[i];
        *sum_d = sum;
    }
}

int main()
{
    cudaSetDevice(0);
    dim3 nthreads(THREADX, 1, 1);
    dim3 nblocks((N + nthreads.x - 1) / nthreads.x, 1, 1);

    printf("nthreads = %d, nblocks = %d\n", nthreads.x, nblocks.x);

    double *x_h = new double[N];
    double *result_h = new double[nblocks.x];
    double sum_h = 0;

    if (x_h == NULL || result_h == NULL)
    {
        printf("Error allocating memory\n");
        exit(1);
    }

    for (auto i = 0; i < N; i++)
        x_h[i] = i * 0.01;

    double *x_d, *result_d, *sum_d;

    cudaMalloc((void **)&x_d, N * sizeof(double));
    cudaMalloc((void **)&result_d, nblocks.x * sizeof(double));
    cudaMalloc((void **)&sum_d, sizeof(double));

    cudaMemcpy(x_d, x_h, N * sizeof(double),
               cudaMemcpyHostToDevice);

    dot_kernel<<<nblocks, nthreads, 0, 0>>>(x_d, result_d);
    cudaDeviceSynchronize();
    /*
      cudaMemcpy(result_h, result_d, nblocks.x * sizeof(double),
               cudaMemcpyDeviceToHost);
      for (auto i = 0; i < nblocks.x; i++)
        printf("result_h[%d] = %f\n", i, result_h[i]);
    */

    sum_kernel<<<1, nthreads, 0, 0>>>(result_d, sum_d);
    cudaDeviceSynchronize();

    cudaMemcpy(&sum_h, sum_d, sizeof(double),
               cudaMemcpyDeviceToHost);
    printf("sum_h = %f\n", sum_h);

    cudaFree(x_d);
    cudaFree(result_d);
    cudaFree(sum_d);
    delete[] x_h;
    delete[] result_h;
}
