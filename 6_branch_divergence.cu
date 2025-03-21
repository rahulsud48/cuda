#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void code_without_divergence()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float a,b;
    a = b = 0;

    int wrap_id = gid / 32;

    if (wrap_id % 2 == 0 )
    {
        a = 100.0;
        b = 50.0;
    }
    else
    {
        a = 50.0;
        b = 100.0;
    }
}

__global__ void divergence_code()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float a,b;
    a = b = 0;

    if (gid % 2 == 0 )
    {
        a = 100.0;
        b = 50.0;
    }
    else
    {
        a = 50.0;
        b = 100.0;
    }
}

int main()
{
    
    int size = 1 << 22;

    dim3 block_size(128);
    dim3 grid_size((size + block_size.x - 1)/ block_size.x);

    code_without_divergence << <grid_size, block_size >> > ();
    cudaDeviceSynchronize();

    divergence_code << <grid_size, block_size >> > ();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}