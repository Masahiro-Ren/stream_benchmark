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
static constexpr size_t ARR_SIZE = 1000;
// static constexpr size_t ARR_SIZE = 1073741824;
static constexpr size_t TO_MB = 1024 * 1024;
static constexpr size_t TO_GB = 1024 * 1024 * 1024;
static constexpr size_t NTIMES = 20;
static constexpr size_t KERNEL_NUM = 4;
static double max_time[KERNEL_NUM] = {0.0};
static double avg_time[KERNEL_NUM] = {0.0};
static double min_time[KERNEL_NUM] = {100.0, 100.0, 100.0, 100.0};
static double total_bytes[KERNEL_NUM] = {
    2.0 * sizeof(STREAM_DATA_TYPE) * ARR_SIZE,
    2.0 * sizeof(STREAM_DATA_TYPE) * ARR_SIZE,
    3.0 * sizeof(STREAM_DATA_TYPE) * ARR_SIZE,
    3.0 * sizeof(STREAM_DATA_TYPE) * ARR_SIZE
};
/**
 * cuda config
 */
static constexpr size_t THREADS = 256;
static constexpr size_t BLOCKS = ((ARR_SIZE + THREADS - 1) / THREADS);

template<typename T>
__global__ void set_array(T *arr, T val, size_t N)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N) arr[idx] = val;
}

template<typename T>
__global__ void stream_copy(T *b, T *c, size_t N)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = b[idx];
}

template<typename T>
__global__ void stream_scale(T *a, T *b, T scalar, size_t N)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
        b[idx] = scalar * a[idx];
}

template<typename T>
__global__ void stream_add(T *a, T *b, T *c, size_t N)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = a[idx] + b[idx];
}

template<typename T>
__global__ void stream_triad(T *a, T *b, T *c, T scalar, size_t N)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = a[idx] + scalar * b[idx];
}


int main()
{
    float ms;
    double scalar = 2.0;
    double time[NTIMES][KERNEL_NUM];

    STREAM_DATA_TYPE *d_a, *d_b, *d_c;
    STREAM_DATA_TYPE *h_a, *h_b, *h_c;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /**
     * Allocate memory
     */
    cudaMalloc((void**)&d_a, sizeof(STREAM_DATA_TYPE) * ARR_SIZE);
    cudaMalloc((void**)&d_b, sizeof(STREAM_DATA_TYPE) * ARR_SIZE);
    cudaMalloc((void**)&d_c, sizeof(STREAM_DATA_TYPE) * ARR_SIZE);

    h_a = new STREAM_DATA_TYPE[ARR_SIZE];
    h_b = new STREAM_DATA_TYPE[ARR_SIZE];
    h_c = new STREAM_DATA_TYPE[ARR_SIZE];

    dim3 dimBlock(BLOCKS);
    dim3 dimGrid(ARR_SIZE/dimBlock.x);

    if(ARR_SIZE % dimBlock.x != 0) dimGrid.x += 1;

    // Init arrays on device
    set_array<STREAM_DATA_TYPE><<<dimGrid, dimBlock>>>(d_a, 4.0, ARR_SIZE);
    set_array<STREAM_DATA_TYPE><<<dimGrid, dimBlock>>>(d_b, 5.0, ARR_SIZE);
    set_array<STREAM_DATA_TYPE><<<dimGrid, dimBlock>>>(d_c, 0.0, ARR_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, ARR_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, ARR_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, ARR_SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(size_t i = 0; i < ARR_SIZE; i++)
    {
        if(h_a[i] != 4.0)
        {
            cout << "wrong value in a" << endl;
            break;
        }
    }
    for(size_t i = 0; i < ARR_SIZE; i++)
    {
        if(h_b[i] != 5.0)
        {
            cout << "wrong value in b" << endl;
            break;
        }
    }
    for(size_t i = 0; i < ARR_SIZE; i++)
    {
        if(h_c[i] != 0.0)
        {
            cout << "wrong value in c" << endl;
            break;
        }
    }

    // // Run test
    // for(size_t n = 0; n < NTIMES; n++)
    // {
    //     cudaEventRecord(start);
    //     stream_copy<STREAM_DATA_TYPE><<<dimGrid, dimBlock>>>(d_b, d_c, ARR_SIZE);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop); 
    //     cudaEventElapsedTime(&ms, start, stop);
    //     time[n][COPY] = ms * 1.0E-3;

    //     cudaEventRecord(start);
    //     stream_scale<STREAM_DATA_TYPE><<<dimGrid, dimBlock>>>(d_a, d_b, scalar, ARR_SIZE);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop); 
    //     cudaEventElapsedTime(&ms, start, stop);
    //     time[n][SCALE] = ms * 1.0E-3;

    //     cudaEventRecord(start);
    //     stream_add<STREAM_DATA_TYPE><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, ARR_SIZE);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop); 
    //     cudaEventElapsedTime(&ms, start, stop);
    //     time[n][ADD] = ms * 1.0E-3;

    //     cudaEventRecord(start);
    //     stream_triad<STREAM_DATA_TYPE><<<dimGrid, dimBlock>>>(d_a, d_b, d_c, scalar, ARR_SIZE);
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop); 
    //     cudaEventElapsedTime(&ms, start, stop);
    //     time[n][TRIAD] = ms * 1.0E-3;
    // }

    // // Summary
    // for(size_t n = 0; n < NTIMES; n++)
    // {
    //     for(size_t k = 0; k < KERNEL_NUM; k++)
    //     {
    //         max_time[k] = std::max(max_time[k], time[n][k]);
    //         min_time[k] = std::min(min_time[k], time[n][k]);
    //         avg_time[k] += time[n][k];
    //     }
    // }

    // for(size_t k = 0; k < KERNEL_NUM; k++)
    // {
    //     total_bytes[k] = total_bytes[k] / min_time[k];
    //     avg_time[k] /= NTIMES;
    // }

    // cout << "Size per array: " << (ARR_SIZE * sizeof(STREAM_DATA_TYPE)) / TO_MB << " MB" << endl;
    // cout << "Size per element: " << (sizeof(STREAM_DATA_TYPE)) << " bytes" << endl;
    // // cout << "Kernel\t" << "Best rate(MB/s)\t" << "Max Time\t" << "Min Time\t" << "Avg Time\t" << endl;
    // cout << "Kernel\t" << "Best rate(GB/s)\t" << "Max Time\t" << "Min Time\t" << "Avg Time\t" << endl;
    // cout << "Copy\t" << total_bytes[COPY] / TO_GB << "\t\t"
    //                  << max_time[COPY] << "\t"
    //                  << min_time[COPY] << "\t"
    //                  << avg_time[COPY] << endl;
    // cout << "Sacle\t" << total_bytes[SCALE] / TO_GB << "\t\t"
    //                   << max_time[SCALE] << "\t"
    //                   << min_time[SCALE] << "\t"
    //                   << avg_time[SCALE] << endl;
    // cout << "Add\t" << total_bytes[ADD] / TO_GB << "\t\t"
    //                 << max_time[ADD] << "\t"
    //                 << min_time[ADD] << "\t"
    //                 << avg_time[ADD] << endl;
    // cout << "Triad\t" << total_bytes[TRIAD] / TO_GB << "\t\t"
    //                   << max_time[TRIAD] << "\t"
    //                   << min_time[TRIAD] << "\t"
    //                   << avg_time[TRIAD] << endl;

    // // Clean up
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}