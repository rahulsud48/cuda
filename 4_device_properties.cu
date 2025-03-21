#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

void query_device()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("The number of devices found : %d \n", deviceCount);

    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties (&prop, device);

    printf("Device %d: %s\n", device, prop.name);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Clock rate: %d\n", prop.clockRate);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total amount of global memory: %.2f KB\n", prop.totalGlobalMem / 1024.0);
    printf("Total amount of constant memory: %.2f KB\n", prop.totalConstMem / 1024.0);
    printf("Total amount of shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Total amount of shared memory per multiprocessor: %.2f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Total number of registers available per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("Maximum Grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

}

int main()
{
    query_device();
    return 0;
}