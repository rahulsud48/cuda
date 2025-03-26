#include <cstdio>
#include <cuda.h>
#include <mma.h>
using namespace nvcuda;

#define M 16
#define N 16
#define K 16

// Kernel using WMMA API for tensor core matrix multiplication
__global__ void wmmaGemmKernel(half *a, half *b, float *c, int lda, int ldb, int ldc) {
    // Each warp computes one 16x16 tile
    // Get warp indices (here we assume one warp per tile)
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // row tile index
    int warpN = blockIdx.y; // column tile index

    // Pointers to the beginning of the tile for A and B
    int aRow = warpM * M;
    int bCol = warpN * N;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, M, N, K, float> cFrag;

    // Initialize output fragment to 0
    wmma::fill_fragment(cFrag, 0.0f);

    // Loop over the K dimension
    for (int k = 0; k < K; k += K) {
        // Load a 16x16 tile of A and B from global memory into fragments
        wmma::load_matrix_sync(aFrag, a + aRow * lda + k, lda);
        wmma::load_matrix_sync(bFrag, b + k * ldb + bCol, ldb);
        // Multiply and accumulate: cFrag = aFrag * bFrag + cFrag
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    // Store the resulting tile back to global memory
    wmma::store_matrix_sync(c + aRow * ldc + bCol, cFrag, ldc, wmma::mem_row_major);
}

int main() {
    // Matrix dimensions (must be multiples of 16)
    int m_total = M, n_total = N, k_total = K;
    int lda = k_total, ldb = n_total, ldc = n_total;
    size_t sizeA = m_total * k_total * sizeof(half);
    size_t sizeB = k_total * n_total * sizeof(half);
    size_t sizeC = m_total * n_total * sizeof(float);

    // Allocate host memory
    half *h_A = (half*)malloc(sizeA);
    half *h_B = (half*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize matrices A and B to 1.0
    for (int i = 0; i < m_total * k_total; i++) {
        h_A[i] = __float2half(1.0f);
    }
    for (int i = 0; i < k_total * n_total; i++) {
        h_B[i] = __float2half(1.0f);
    }
    // Initialize matrix C to zero
    for (int i = 0; i < m_total * n_total; i++) {
        h_C[i] = 0.0f;
    }

    // Allocate device memory
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice);

    // Launch kernel:
    // Since WMMA operations require one warp (32 threads) per tile, and our matrices are 16x16, we need only one warp.
    dim3 gridDim(1, 1);
    dim3 blockDim(32, 1);
    wmmaGemmKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, lda, ldb, ldc);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print the output matrix
    printf("Result matrix C:\n");
    for (int i = 0; i < m_total; i++) {
        for (int j = 0; j < n_total; j++) {
            printf("%f ", h_C[i * n_total + j]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
