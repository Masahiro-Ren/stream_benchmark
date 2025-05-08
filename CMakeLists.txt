cmake_minimum_required(VERSION 3.16)
project(KOKKOS_STREAM LANGUAGES CXX)

# Find Kokkos package
set(Kokkos_ROOT "/home/kokkos/kokkos-4.6")
set(Kokkos_DIR "/home/kokkos/kokkos-4.6/lib/cmake")
find_package(Kokkos REQUIRED)

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)

# Add executable
add_executable(kokkos_stream kokkos_stream.cpp)

# Link Kokkos to your executable
target_link_libraries(kokkos_stream Kokkos::kokkos)

# (Optional) enable C++17 if your compiler needs explicit setting
target_compile_options(kokkos_stream PRIVATE -Wall -O3 -fopenmp -march=native)
target_compile_features(kokkos_stream PRIVATE cxx_std_17)
