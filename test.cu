#include <iostream>

template<typename T>
__global__ void set_array(T *arr, T val, size_t N)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if()
}

int main()
{
    return 0;
}