#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>

__global__ void print_details()
{
    int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    int wrap_id = threadIdx.x / 32;

    int gbid = blockIdx.y * gridDim.x + blockIdx.x;

    printf("tid : %d, bid.x : % d, bid.y : %d, gid : %d, wrap_id : %d, gbid.y : %d \n", 
            threadIdx.x, blockIdx.x, blockIdx.y, gid, wrap_id, gbid);
}


int main()
{
    dim3 block(42);
    dim3 grid(2,2);

    print_details << <grid,block>> > ();

    cudaDeviceSynchronize();

    cudaDeviceReset();
}