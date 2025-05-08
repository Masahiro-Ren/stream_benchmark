#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <iostream>
#include <chrono>

#define USE_SIMD

using std::cout;
using std::endl;
using timer = std::chrono::high_resolution_clock;
using GET_DURATION = std::chrono::duration<double, std::micro>;

using Kokkos::View;
using Kokkos::RangePolicy;
using Kokkos::OpenMP;
using Kokkos::Schedule;

using REAL = double;

static constexpr size_t N = 100000000;

#ifndef USE_SIMD
using DATA_TYPE = REAL;
static constexpr size_t VL = 1;
#else
using DATA_TYPE = Kokkos::Experimental::simd<REAL>;
static constexpr size_t VL = DATA_TYPE::size();
#endif

static constexpr size_t ARR_SIZE = (N + VL - 1) / VL;
static constexpr size_t TO_MB = 1024 * 1024;
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
    2.0 * sizeof(REAL) * N,
    2.0 * sizeof(REAL) * N,
    3.0 * sizeof(REAL) * N,
    3.0 * sizeof(REAL) * N
};


struct copy {

    copy(View<DATA_TYPE*> a_, View<DATA_TYPE*> c_): a(a_), c(c_) {};

    KOKKOS_FUNCTION
    void operator()(size_t i) const
    {
        c(i) = a(i);
    }
private:
    View<DATA_TYPE*> a;
    View<DATA_TYPE*> c;
};

struct scale {
    scale(View<DATA_TYPE*> b_, View<DATA_TYPE*> c_, REAL s): b(b_), c(c_), scalar(s) {};

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const
    {
        b(i) = scalar * c(i);
    }
private:
    View<DATA_TYPE*> b;
    View<DATA_TYPE*> c;
    REAL scalar;
};

struct add {
    add(View<DATA_TYPE*> a_, View<DATA_TYPE*> b_, View<DATA_TYPE*> c_): a(a_), b(b_), c(c_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const
    {
        c(i) = a(i) + b(i);
    }
private:
    View<DATA_TYPE*> a;
    View<DATA_TYPE*> b;
    View<DATA_TYPE*> c;
};

struct triad {
    triad(View<DATA_TYPE*> a_, View<DATA_TYPE*> b_, View<DATA_TYPE*> c_, REAL s): a(a_), b(b_), c(c_), scalar(s) {};

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const
    {
        a(i) = b(i) + scalar * c(i);
    }
private:
    View<DATA_TYPE*> a;
    View<DATA_TYPE*> b;
    View<DATA_TYPE*> c;
    REAL scalar;
};

int main(int argc, char* argv[])
{
Kokkos::initialize(argc, argv);
{
    REAL scalar = 2.0;
    View<DATA_TYPE*> a("a", ARR_SIZE);
    View<DATA_TYPE*> b("b", ARR_SIZE);
    View<DATA_TYPE*> c("c", ARR_SIZE);

    // Init
    cout << "VL: " << VL << endl;

    double time[NTIMES][KERNEL_NUM];

    Kokkos::parallel_for(RangePolicy<OpenMP, Schedule<Kokkos::Static>>(0, ARR_SIZE), 
    KOKKOS_LAMBDA(const size_t i){
        a(i) = 4.0;
        b(i) = 5.0;
        c(i) = 0.0;
    });

    for(size_t n = 0; n < NTIMES; n++)
    {
        auto copy_st = timer::now();
        Kokkos::parallel_for(RangePolicy<OpenMP, Schedule<Kokkos::Static>>(0, ARR_SIZE), copy(a, c));
        auto copy_ed = timer::now();

        auto scale_st = timer::now();
        Kokkos::parallel_for(RangePolicy<OpenMP, Schedule<Kokkos::Static>>(0, ARR_SIZE), scale(b, c, scalar));
        auto scale_ed = timer::now();

        auto add_st = timer::now();
        Kokkos::parallel_for(RangePolicy<OpenMP, Schedule<Kokkos::Static>>(0, ARR_SIZE), add(a, b, c));
        auto add_ed = timer::now();

        auto triad_st = timer::now();
        Kokkos::parallel_for(RangePolicy<OpenMP, Schedule<Kokkos::Static>>(0, ARR_SIZE), triad(a, b, c, scalar));
        auto triad_ed = timer::now();

        double t_copy  = GET_DURATION(copy_ed - copy_st).count();
        double t_scale = GET_DURATION(scale_ed - scale_st).count();
        double t_add   = GET_DURATION(add_ed - add_st).count();
        double t_triad = GET_DURATION(triad_ed - triad_st).count();

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

    cout << "Size per array: " << (N * sizeof(REAL)) / TO_MB << " MB" << endl;
    cout << "Size per element: " << (sizeof(REAL)) << " bytes" << endl;
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
}
Kokkos::finalize();
}