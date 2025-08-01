#include "tensor.hpp"          // Tensor<T, Rank>
#include "einsum.hpp"
#include <gtest/gtest.h>
#include <random>
#include <chrono>

/* ── helpers ── */
template<class TensorLike>
void fill_random(TensorLike& t, unsigned seed = 1234)
{
    using value_t = typename TensorLike::value_type;
    static_assert(std::is_floating_point<value_t>::value,
                  "fill_random expects a floating-point tensor");

    std::mt19937                       rng(seed);
    std::uniform_real_distribution<value_t> dist(value_t(-1.0), value_t(1.0));

    for (std::size_t i = 0; i < t.size(); ++i)
        t._data()[i] = dist(rng);
}

TEST(Speed, BatchedMatMul_300x300x300)
{
    /* A : [8,300,300]   B : [8,300,300]
       C = "bmk,bkn->bmn"  (≈216 M FLOPs) */
    using T   = float;
    using Ten = Tensor<T, 3>;

    Ten A({{8, 300, 300}});
    Ten B({{8, 300, 300}});
    fill_random(A);
    fill_random(B);

    auto t0 = std::chrono::high_resolution_clock::now();
    Ten  C  = einsum<T, 3>("bmk,bkn->bmn", {A, B});   // ← rank is 3
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "300×300×300 batched matmul took "
              << ms << " ms\n";

    EXPECT_LT(ms, 60000.0);     // generous Debug limit
}
