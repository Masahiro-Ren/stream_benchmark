cmake_minimum_required(VERSION 3.16)
project(KokkosApp LANGUAGES CXX)

# Choose C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optionally allow CMake to find CUDA/HIP
find_package(Kokkos REQUIRED)

# Add your single source file
add_executable(kokkos_stream kokkos_stream.cpp)

# Link Kokkos targets (this adds include dirs, flags, etc.)
target_link_libraries(kokkos_stream Kokkos::kokkos)

# Optional: print target compile flags (useful for debugging)
# get_target_property(FLAGS kokkos_app COMPILE_OPTIONS)
# message(STATUS "Compile Flags: ${FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fast -mp")
message(STATUS "Compile with CXX Flags: ${CMAKE_CXX_FLAGS}")

