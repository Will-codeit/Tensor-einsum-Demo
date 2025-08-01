#include "tensor.hpp"      // Tensor<T, Rank>
#include "einsum.hpp"
#include <iostream>

int main()
{
    using Ten1 = Tensor<double, 1>;      // rank-1 tensor (vector)

    Ten1 a({{3}}, 1.0);                  // [1,1,1]
    Ten1 b({{3}});                       // [?, ?, ?]

    for (std::size_t i = 0; i < 3; ++i)
        b({{i}}) = static_cast<double>(i + 2);   // 2,3,4

    Ten1 dot = einsum<double, 1>("i,i->", {a, b});
    std::cout << "dot = " << dot({{}}) << std::endl;   // prints 9
    return 0;
}
