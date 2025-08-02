#include "tensor.hpp"
#include "einsum.hpp"
#include <gtest/gtest.h>

/* ───────── helpers ────────── */
TEST(intensive, matmul_accuracy_test)
{
    /* A : [8,300,300]   B : [8,300,300]
       C = "bmk,bkn->bmn"  (≈216 M FLOPs) */
    using T   = int;
    using Ten = Tensor<T, 2>;

    Ten A({3, 2}, {0, -1, 0, 0, 0, -1});
    Ten B({2, 3}, {1, 0, 0, -1, 1, 0});

    Ten C  = einsum<T, 2>("bk,kn->bn", {A, B});
    std::cout << "where am i printing?" << endl << endl << endl;
    std::cout << "result = "
              << C.listToString(C.getData());
}
/*
C should be : (array([ 1, -1,  0,  0,  0,  0,  1, -1,  0]),
A is : array([ 0, -1,  0,  0,  0, -1]),
B is : array([ 1,  0,  0, -1,  1,  0]))

checked on numpy
 */
