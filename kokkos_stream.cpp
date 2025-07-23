#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;
using timer = std::chrono::high_resolution_clock;
using GET_DURATION = std::chrono::duration<double, std::micro>;

using Kokkos::View;
using Kokkos::RangePolicy;
using Kokkos::Schedule;

using DATA_TYPE = float;
using HOST_SPACE = Kokkos::OpenMP;
using EXE_SPACE = typename Kokkos::DefaultExecutionSpace;
using MEM_SPACE = typename Kokkos::DefaultExecutionSpace::memory_space;


// static constexpr size_t ARR_SIZE = 120000000;
static constexpr size_t ARR_SIZE = 1071741824;
static constexpr size_t TO_MB = 1024 * 1024;
static constexpr size_t NTIMES = 200;
static constexpr size_t KERNEL_NUM = 4;

enum {
    COPY,
    SCALE,
    ADD,
    TRIAD
};

static double max_time[KERNEL_NUM] = {0.0};
static double min_time[KERNEL_NUM] = {999.0, 999.0, 999.0, 999.0};
static double avg_time[KERNEL_NUM] = {0.0};
static double total_bytes[KERNEL_NUM] = {
    2.0 * ARR_SIZE * sizeof(DATA_TYPE),
    2.0 * ARR_SIZE * sizeof(DATA_TYPE),
    3.0 * ARR_SIZE * sizeof(DATA_TYPE),
    3.0 * ARR_SIZE * sizeof(DATA_TYPE)
};


struct copy {

    copy(View<DATA_TYPE*, MEM_SPACE> a_, View<DATA_TYPE*, MEM_SPACE> c_): a(a_), c(c_) {};

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
    scale(View<DATA_TYPE*, MEM_SPACE> b_, View<DATA_TYPE*, MEM_SPACE> c_, DATA_TYPE s): b(b_), c(c_), scalar(s) {};

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const
    {
        b(i) = scalar * c(i);
    }
private:
    View<DATA_TYPE*> b;
    View<DATA_TYPE*> c;
    DATA_TYPE scalar;
};

struct add {
    add(View<DATA_TYPE*, MEM_SPACE> a_, View<DATA_TYPE*, MEM_SPACE> b_, View<DATA_TYPE*, MEM_SPACE> c_): a(a_), b(b_), c(c_) {}

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
    triad(View<DATA_TYPE*, MEM_SPACE> a_, View<DATA_TYPE*, MEM_SPACE> b_, View<DATA_TYPE*, MEM_SPACE> c_, DATA_TYPE s): a(a_), b(b_), c(c_), scalar(s) {};

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t i) const
    {
        a(i) = b(i) + scalar * c(i);
    }
private:
    View<DATA_TYPE*> a;
    View<DATA_TYPE*> b;
    View<DATA_TYPE*> c;
    DATA_TYPE scalar;
};

int main(int argc, char* argv[])
{
Kokkos::initialize(argc, argv);
{
    DATA_TYPE scalar = 2.0;
    View<DATA_TYPE*, MEM_SPACE> a("a", ARR_SIZE);
    View<DATA_TYPE*, MEM_SPACE> b("b", ARR_SIZE);
    View<DATA_TYPE*, MEM_SPACE> c("c", ARR_SIZE);

    // Init
    double time[NTIMES][KERNEL_NUM];

    auto h_a = Kokkos::create_mirror_view(a);
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_c = Kokkos::create_mirror_view(c);

    Kokkos::parallel_for(RangePolicy<HOST_SPACE, Schedule<Kokkos::Static>>(0, ARR_SIZE), 
    KOKKOS_LAMBDA(const size_t i){
        h_a(i) = 4.0;
        h_b(i) = 5.0;
        h_c(i) = 0.0;
    });

    // No op when host memory space
    Kokkos::deep_copy(a, h_a);
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(c, h_c);
    Kokkos::fence();

    Kokkos::Timer kks_timer;

    for(size_t n = 0; n < NTIMES; n++)
    {
        // auto copy_st = timer::now();
        kks_timer.reset();
        Kokkos::parallel_for(RangePolicy<EXE_SPACE, Schedule<Kokkos::Static>>(0, ARR_SIZE), copy(a, c));
        Kokkos::fence();
        time[n][COPY] = kks_timer.seconds(); 
        // auto copy_ed = timer::now();

        // auto scale_st = timer::now();
        kks_timer.reset();
        Kokkos::parallel_for(RangePolicy<EXE_SPACE, Schedule<Kokkos::Static>>(0, ARR_SIZE), scale(b, c, scalar));
        Kokkos::fence();
        time[n][SCALE] = kks_timer.seconds();
        // auto scale_ed = timer::now();

        // auto add_st = timer::now();
        kks_timer.reset();
        Kokkos::parallel_for(RangePolicy<EXE_SPACE, Schedule<Kokkos::Static>>(0, ARR_SIZE), add(a, b, c));
        Kokkos::fence();
        time[n][ADD] = kks_timer.seconds();
        // auto add_ed = timer::now();

        // auto triad_st = timer::now();
        kks_timer.reset();
        Kokkos::parallel_for(RangePolicy<EXE_SPACE, Schedule<Kokkos::Static>>(0, ARR_SIZE), triad(a, b, c, scalar));
        Kokkos::fence();
        time[n][TRIAD] = kks_timer.seconds();
        // auto triad_ed = timer::now();

        // double t_copy  = GET_DURATION(copy_ed - copy_st).count();
        // double t_scale = GET_DURATION(scale_ed - scale_st).count();
        // double t_add   = GET_DURATION(add_ed - add_st).count();
        // double t_triad = GET_DURATION(triad_ed - triad_st).count();

        // // Mircro sec. to Sec.
        // time[n][COPY]  = t_copy  * 1.0E-6;    
        // time[n][SCALE] = t_scale * 1.0E-6;    
        // time[n][ADD]   = t_add   * 1.0E-6;    
        // time[n][TRIAD] = t_triad * 1.0E-6;  
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

    cout << "Default memroy Space: " << typeid(MEM_SPACE).name() << std::endl;
    cout << "Size per array: " << (ARR_SIZE * sizeof(DATA_TYPE)) / TO_MB << " MB" << endl;
    cout << "Size per element: " << (sizeof(DATA_TYPE)) << " bytes" << endl;
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
