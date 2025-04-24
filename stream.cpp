#include <iostream>
#include <new>
#include <omp.h>
#include <chrono>

using std::cout;
using std::endl;
using timer = std::chrono::high_resolution_clock;
using GET_DURARION = std::chrono::duration<double, std::micro>;

using STREAM_DATA_TYPE = double;

static constexpr size_t ARR_SIZE = 200000000;
static constexpr size_t TO_MB = 1024 * 1024;
static constexpr STREAM_DATA_TYPE scalar = 2.0;
static constexpr size_t NTIMES = 20;
static constexpr size_t KERNEL_NUM = 4;

enum {
    COPY,
    SCALE,
    ADD,
    TRIAD
};
static double max_time[KERNEL_NUM] = {0.0};
static double min_time[KERNEL_NUM] = {100.0, 100.0, 100.0, 100.0};
static double avg_time[KERNEL_NUM] = {0.0};
static double total_bytes[KERNEL_NUM] = {
    2.0 * sizeof(STREAM_DATA_TYPE) * ARR_SIZE,
    2.0 * sizeof(STREAM_DATA_TYPE) * ARR_SIZE,
    3.0 * sizeof(STREAM_DATA_TYPE) * ARR_SIZE,
    3.0 * sizeof(STREAM_DATA_TYPE) * ARR_SIZE
};

void copy(STREAM_DATA_TYPE a[], STREAM_DATA_TYPE c[])
{
    #pragma omp parallel for simd
    for(size_t i = 0; i < ARR_SIZE; i++)
        c[i] = a[i];
} 

void scale(STREAM_DATA_TYPE b[], STREAM_DATA_TYPE c[])
{
    #pragma omp parallel for simd
    for(size_t i = 0; i < ARR_SIZE; i++)
        b[i] = scalar * c[i];    
}

void add(STREAM_DATA_TYPE a[], STREAM_DATA_TYPE b[], STREAM_DATA_TYPE c[])
{
    #pragma omp parallel for simd
    for(size_t i = 0; i < ARR_SIZE; i++)
        c[i] = a[i] + b[i];
}

void triad(STREAM_DATA_TYPE a[], STREAM_DATA_TYPE b[], STREAM_DATA_TYPE c[])
{
    #pragma omp parallel for simd
    for(size_t i = 0; i < ARR_SIZE; i++)
        a[i] = b[i] + scalar * c[i];
}

int main()
{
    // STREAM_DATA_TYPE *a = new STREAM_DATA_TYPE[ARR_SIZE];
    // STREAM_DATA_TYPE *b = new STREAM_DATA_TYPE[ARR_SIZE];
    // STREAM_DATA_TYPE *c = new STREAM_DATA_TYPE[ARR_SIZE];
    STREAM_DATA_TYPE *a = new (std::align_val_t(64))STREAM_DATA_TYPE[ARR_SIZE];
    STREAM_DATA_TYPE *b = new (std::align_val_t(64))STREAM_DATA_TYPE[ARR_SIZE];
    STREAM_DATA_TYPE *c = new (std::align_val_t(64))STREAM_DATA_TYPE[ARR_SIZE];

    double time[NTIMES][KERNEL_NUM];

    // Init
    #pragma omp parallel for simd
    for(size_t i = 0; i < ARR_SIZE; i++)
    {
        a[i] = 4.0;
        b[i] = 5.0;
        c[i] = 0.0;
    }

    // Run test
    for(size_t n = 0; n < NTIMES; n++)
    {
        auto copy_st = timer::now();
        copy(a, c);
        auto copy_ed = timer::now();

        auto scale_st = timer::now();
        scale(b, c);
        auto scale_ed = timer::now();

        auto add_st = timer::now();
        add(a, b, c);
        auto add_ed = timer::now();

        auto triad_st = timer::now();
        triad(a, b, c);
        auto triad_ed = timer::now();

        double t_copy  = GET_DURARION(copy_ed - copy_st).count();
        double t_scale = GET_DURARION(scale_ed - scale_st).count();
        double t_add   = GET_DURARION(add_ed - add_st).count();
        double t_triad = GET_DURARION(triad_ed - triad_st).count();

        // Mircro sec. to Sec.
        time[n][COPY]  = t_copy  * 1.0E-6;    
        time[n][SCALE] = t_scale * 1.0E-6;    
        time[n][ADD]   = t_add   * 1.0E-6;    
        time[n][TRIAD] = t_triad * 1.0E-6;    
    }

    // Summary
    for(size_t n = 0; n < NTIMES; n++)
    {
        for(size_t k = 0; k < KERNEL_NUM; k++)
        {
            max_time[k] = std::max(max_time[k], time[n][k]);
            min_time[k] = std::min(min_time[k], time[n][k]);
            avg_time[k] += time[n][k];
        }
    }

    for(size_t k = 0; k < KERNEL_NUM; k++)
    {
        total_bytes[k] = total_bytes[k] / min_time[k];
        avg_time[k] /= NTIMES;
    }

    cout << "Size per array: " << (ARR_SIZE * sizeof(STREAM_DATA_TYPE)) / TO_MB << " MB" << endl;
    cout << "Size per element: " << (sizeof(STREAM_DATA_TYPE)) << " bytes" << endl;
    cout << "Kernel\t" << "Best rate(MB/s)\t" << "Max Time\t" << "Min Time\t" << "Avg Time\t" << endl;
    cout << "Copy\t" << total_bytes[COPY] / TO_MB << "\t\t"
                     << max_time[COPY] << "\t"
                     << min_time[COPY] << "\t"
                     << avg_time[COPY] << endl;
    cout << "Sacle\t" << total_bytes[SCALE] / TO_MB << "\t\t"
                      << max_time[SCALE] << "\t"
                      << min_time[SCALE] << "\t"
                      << avg_time[SCALE] << endl;
    cout << "Add\t" << total_bytes[ADD] / TO_MB << "\t\t"
                    << max_time[ADD] << "\t"
                    << min_time[ADD] << "\t"
                    << avg_time[ADD] << endl;
    cout << "Triad\t" << total_bytes[TRIAD] / TO_MB << "\t\t"
                      << max_time[TRIAD] << "\t"
                      << min_time[TRIAD] << "\t"
                      << avg_time[TRIAD] << endl;

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}