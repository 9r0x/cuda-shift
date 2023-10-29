#include "utils.h"

__host__ void catch_error(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[!] CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}