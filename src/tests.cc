#include <iostream>
#include <cassert>
#include <random>

#include "b_tree.h"
#include "sampled_b_tree.h"
#include "ebs.h"
#include "ips4o.hpp"

#ifdef TEST_INT32
typedef int32_t data_t;
#elif defined TEST_INT64
typedef int64_t data_t;
#elif defined TEST_UINT32
typedef uint32_t data_t;
#elif defined TEST_UINT64
typedef uint64_t data_t;
#elif defined TEST_DOUBLE
typedef double data_t;
#elif defined TEST_FLOAT
typedef float data_t;
#endif

constexpr uint_fast32_t seed = 42;

template <typename Index>
void test(const data_t * v, size_t sz, std::string test_name) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> distrib(0, std::numeric_limits<data_t>::max());
    
    Index e;
    e.build(v, v + sz);

    // Search al the keys
    for (size_t i = 0; i < sz; ++i) {
        auto search_res = e.search(v[i]);
        if (v[i] != v[search_res])
            throw std::runtime_error(test_name + " TEST FAIL: Existing key test error, expected: " + std::to_string(i) + " obtained: " + std::to_string(search_res));
    }
    
    // Search non exixting keys
    constexpr size_t n_tests = (size_t) 1e6;
    for (size_t i = 0; i < n_tests; ++i) {
        data_t key = distrib(gen);

        auto search_res = e.search(key);
        auto std_res = std::lower_bound(v, v + sz, key);

        if (*std_res != v[search_res])
            throw std::runtime_error(test_name + " TEST FAIL: Non existing key test error, expected: " + std::to_string(std_res - v) + " obtained: " + std::to_string(search_res));
    }
}

void many_tests(const size_t sz) {
    data_t * v = (data_t*)std::aligned_alloc(64, sizeof(data_t) * sz);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> distrib(std::numeric_limits<data_t>::lowest(), std::numeric_limits<data_t>::max());

    std::generate(v, v + sz, [&](){return distrib(gen); });
    ips4o::parallel::sort(v, v + sz);

    // EBS
    test<searches::ebs<data_t>>(v, sz, "EBS" + std::to_string(sz));

    // B-Tree
    test<SIMD_Btree::btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 1>>(v, sz, "EBS line avx2 1 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 2>>(v, sz, "EBS line avx2 2 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 4>>(v, sz, "EBS line avx2 4 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 1>>(v, sz, "EBS line avx512 1 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 2>>(v, sz, "EBS line avx512 2 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 4>>(v, sz, "EBS line avx512 4 " + std::to_string(sz));

    // Sampled B-Tree
    test<SIMD_Btree::b_plus_tree<data_t, SIMD_Btree::SIMD_ext::AVX2, 1>>(v, sz, "B PLUS avx2 1 " + std::to_string(sz));
    test<SIMD_Btree::b_plus_tree<data_t, SIMD_Btree::SIMD_ext::AVX2, 2>>(v, sz, "B PLUS avx2 2 " + std::to_string(sz));
    test<SIMD_Btree::b_plus_tree<data_t, SIMD_Btree::SIMD_ext::AVX2, 4>>(v, sz, "B PLUS avx2 4 " + std::to_string(sz));
    test<SIMD_Btree::b_plus_tree<data_t, SIMD_Btree::SIMD_ext::AVX512, 1>>(v, sz, "B PLUS avx512 1 " + std::to_string(sz));
    test<SIMD_Btree::b_plus_tree<data_t, SIMD_Btree::SIMD_ext::AVX512, 2>>(v, sz, "B PLUS avx512 2 " + std::to_string(sz));
    test<SIMD_Btree::b_plus_tree<data_t, SIMD_Btree::SIMD_ext::AVX512, 4>>(v, sz, "B PLUS avx512 4 " + std::to_string(sz));
    
    // B+ tree with leaves of different size
    test<SIMD_Btree::b_plus_tree<data_t, SIMD_Btree::SIMD_ext::AVX2, 2, 4>>(v, sz, "B PLUS avx2 2-4 " + std::to_string(sz));
    test<SIMD_Btree::b_plus_tree<data_t, SIMD_Btree::SIMD_ext::AVX512, 4, 1>>(v, sz, "B PLUS avx512 4-1 " + std::to_string(sz));
    
    std::free(v);
}

int main() {
    many_tests(2);
    many_tests(8);
    many_tests(10);
    many_tests(16);
    many_tests(32);
    many_tests(63);
    many_tests(64);
    many_tests(65);
    many_tests(1000);
    many_tests(100'000);
    many_tests(std::pow(8,4));
    many_tests(std::pow(9,4));
    many_tests(1'000'000);
    
    std::cout << "Tests OK" << std::endl;
    return 0;
}
