#include "tensor.hpp"
#include "einsum.hpp"
#include <iostream>

int main()
{
    Tensor<double> a({3}, 1.0);
    Tensor<double> b({3});
    for (size_t i = 0; i < 3; ++i) b({i}) = i + 2;   

    Tensor<double> dot = einsum<double>("i,i->", {a, b});
    std::cout << "dot = " << dot({}) << std::endl;  
    return 0;
}