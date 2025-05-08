#include <cuda_runtime.h>
#include <iostream>

#define N (201326592)  // 768 MB per array (201,326,592 floats)
#define THREADS_PER_BLOCK 512
#define SCALAR 3.0f

__global__ void copyKernel(float *A, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i];
}

__global__ void scaleKernel(float *B, float *C, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        B[i] = scalar * C[i];
}

__global__ void addKernel(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

__global__ void triadKernel(float *A, float *B, float *C, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        A[i] = B[i] + scalar * C[i];
}

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void reportTimeAndBandwidth(float ms, size_t bytes, const char* label) {
    double bandwidth = (bytes / (ms / 1000.0)) / (1 << 30);
    std::cout << label << ": " << ms << " ms, " << bandwidth << " GB/s" << std::endl;
}

int main() {

    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;
    size_t size = N * sizeof(float);
    size_t blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_C[i] = 0.0f;
    }

    // Allocate device memory
    checkCuda(cudaMalloc(&d_A, size), "cudaMalloc A");
    checkCuda(cudaMalloc(&d_B, size), "cudaMalloc B");
    checkCuda(cudaMalloc(&d_C, size), "cudaMalloc C");

    // Copy initialized data to device
    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Memcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Memcpy B");
    checkCuda(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice), "Memcpy C");

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float msCopy, msScale, msAdd, msTriad;

    // Copy
    cudaEventRecord(start);
    copyKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msCopy, start, stop);
    reportTimeAndBandwidth(msCopy, 2 * size, "Copy");

    // Scale
    cudaEventRecord(start);
    scaleKernel<<<blocks, THREADS_PER_BLOCK>>>(d_B, d_C, SCALAR, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msScale, start, stop);
    reportTimeAndBandwidth(msScale, 2 * size, "Scale");

    // Add
    cudaEventRecord(start);
    addKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msAdd, start, stop);
    reportTimeAndBandwidth(msAdd, 3 * size, "Add");

    // Triad
    cudaEventRecord(start);
    triadKernel<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, SCALAR, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msTriad, start, stop);
    reportTimeAndBandwidth(msTriad, 3 * size, "Triad");

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
