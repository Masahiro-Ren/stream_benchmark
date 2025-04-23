#include <iostream>
#include <chrono>

using std::cout;
using std::endl;
using timer = std::chrono::high_resolution_clock;
using GET_DURATION = std::chrono::duration<double, std::micro>;

using STREAM_DATA_TYPE = double;

enum {
    COPY,
    SCALE,
    ADD,
    TRIAD
};

/**
 * Stream bench config
 */
static constexpr size_t ARR_SIZE = 100000000;
static constexpr size_t TO_MB = 1024 * 1024;
static constexpr size_t NTIMES = 20;
static constexpr size_t KERNEL_NUM = 4;
static double max_time[KERNEL_NUM] = {0.0};
static double avg_time[KERNEL_NUM] = {0.0};
static double min_time[KERNEL_NUM] = {100.0, 100.0, 100.0, 100.0};
/**
 * cuda config
 */
static constexpr size_t THREADS = 256;
static constexpr size_t BLOCKS = ((ARR_SIZE + THREADS - 1) / THREADS);

__global__ void set_array(STREAM_DATA_TYPE* arr, STREAM_DATA_TYPE val, size_t N)
{
    size_t idx = threadIdx.x + threadIdx.x * blockDim.x;
    if(idx < N) arr[idx] = val;
}

__global__ void stream_copy(STREAM_DATA_TYPE* b, STREAM_DATA_TYPE* c, size_t N)
{
    size_t idx = threadIdx.x + threadIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = b[idx];
}

__global__ void stream_scale(STREAM_DATA_TYPE* a, STREAM_DATA_TYPE* b, STREAM_DATA_TYPE scalar, size_t N)
{
    size_t idx = threadIdx.x + threadIdx.x * blockDim.x;
    if(idx < N)
        b[idx] = scalar * a[idx];
}

__global__ void stream_add(STREAM_DATA_TYPE* a, STREAM_DATA_TYPE* b, STREAM_DATA_TYPE* c, size_t N)
{
    size_t idx = threadIdx.x + threadIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = a[idx] + b[idx];
}

__global__ void stream_triad(STREAM_DATA_TYPE *a, STREAM_DATA_TYPE *b, STREAM_DATA_TYPE *c, STREAM_DATA_TYPE scalar, size_t N)
{
    size_t idx = threadIdx.x + threadIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = a[idx] + scalar * b[idx];
}


int main()
{
    double scalar = 2.0;
    double time[NTIMES][KERNEL_NUM];

    STREAM_DATA_TYPE *d_a, *d_b, *d_c;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /**
     * allocate memory
     */
    cudaMalloc((void**)&d_a, sizeof(STREAM_DATA_TYPE) * ARR_SIZE);
    cudaMalloc((void**)&d_b, sizeof(STREAM_DATA_TYPE) * ARR_SIZE);
    cudaMalloc((void**)&d_c, sizeof(STREAM_DATA_TYPE) * ARR_SIZE);

    dim3 dimBlock(BLOCKS);
    dim3 dimGrid(ARR_SIZE/dimBlock.x);

    if(ARR_SIZE % dimBlock.x != 0) dimGrid.x += 1;

    /**
     * Init arrays
     */
    
    cudaEventRecord(start);
    set_array<<<dimGrid, dimBlock>>>(d_a, 4.0, ARR_SIZE);
    set_array<<<dimGrid, dimBlock>>>(d_b, 5.0, ARR_SIZE);
    set_array<<<dimGrid, dimBlock>>>(d_c, 0.0, ARR_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float m_time;
    cudaEventElapsedTime(&m_time, start, stop);

    cout << "Init time: " << m_time << " ms" << endl;


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}