#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>

__global__ void print_details()
{
    
    printf("threadIdx.x : %d, threadIdx.y : % d, threadIdx.z : %d \n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx.x : %d, blockIdx.y : % d, blockIdx.z : %d \n", blockIdx.x, blockIdx.y, blockIdx.z);
    printf("gridDim.x : %d, gridDim.y : % d, gridDim.z : %d \n", gridDim.x, gridDim.y, gridDim.z);
}


int main()
{

    int nx, ny, nz;
    nx = 8; ny = 8; nz = 16;
    dim3 block(4,4,8);
    dim3 grid(nx/block.x, ny/block.y, nz/block.z);

    print_details << <grid,block>> > ();

    cudaDeviceSynchronize();

    cudaDeviceReset();
}