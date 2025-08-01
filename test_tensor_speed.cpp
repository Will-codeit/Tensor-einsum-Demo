#include "tensor.hpp"
#include "einsum.hpp"
#include <gtest/gtest.h>
#include <random>                
#include <chrono>

/* ── helpers ── */
template<typename T>
void fill_random(Tensor<T>& t, unsigned seed = 1234)
{
    static_assert(std::is_floating_point<T>::value,
                  "fill_random expects a floating-point tensor");

    std::mt19937                    rng(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(-1.0),
                                           static_cast<T>(1.0));

    for (std::size_t i = 0; i < t.size(); ++i)
        t._data()[i] = dist(rng);
}

TEST(Speed, BatchedMatMul_300x300x300)
{
    /*
        A : [8,300,300]   B : [8,300,300]
        C = "bmk,bkn->bmn"   (~216 M mul-adds)
    */
    Tensor<float> A({8, 300, 300});
    Tensor<float> B({8, 300, 300});
    fill_random(A);
    fill_random(B);

    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor<float> C = einsum<float>("bmk,bkn->bmn", {A, B});
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "300×300×300 batched matmul took " << ms << " ms\n";

    EXPECT_LT(ms, 60000.0);   // fail if slower than 10 s
}