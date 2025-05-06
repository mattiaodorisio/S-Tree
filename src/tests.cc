#include <iostream>
#include <cassert>
#include <random>

#include "b_tree.h"
#include "sampled_b_tree.h"
#include "b_tree_pgm.h"
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

template <typename Index, bool approximate = false>
void test(const data_t * v, size_t sz, std::string test_name) {
    Index e(v, v + sz);

    if constexpr (!approximate) {
        if (Index::support_access()) {
            for (size_t i = 0; i < sz; ++i) {
                if (e[i] != v[i])
                    throw std::runtime_error("operator [] test fail at index: " + std::to_string(i));
            }
        }
    }

    // Search al the keys
    for (size_t i = 0; i < sz; ++i) {
        size_t search_res_lower_idx;
        size_t search_res_upper_idx;
        
        // Make the query
        if constexpr (!approximate) {
            search_res_lower_idx = e.lower_bound_idx(v[i]);
            search_res_upper_idx = e.upper_bound_idx(v[i]);
        } else {
            auto search_res = e.search(v[i]);
            search_res_lower_idx = std::distance(v, std::lower_bound(v + search_res.lo, v + search_res.hi, v[i]));
        }

        // Check the query
        if (v[i] != v[search_res_lower_idx])
            throw std::runtime_error(test_name + " TEST FAIL: Lower - Existing key test error, expected: " + std::to_string(i) + " obtained: " + std::to_string(search_res_lower_idx));

        if constexpr (!approximate) {
            if (v[i] != v[search_res_upper_idx - 1])
                throw std::runtime_error(test_name + " TEST FAIL: Upper - Existing key test error, expected: " + std::to_string(i) + " obtained: " + std::to_string(search_res_upper_idx) + ", size: " + std::to_string(sz));
        }
    }

    // Generate non-existing keys
    std::mt19937 gen(seed + 1);
    data_t min_ = std::numeric_limits<data_t>::lowest(), max_ = std::numeric_limits<data_t>::max();
    if (typeid(data_t) == typeid(double)) {
        // Reduce range for doubles, see why here: https://stackoverflow.com/a/66854452
        max_ /= 2;
        min_ = -max_;
    }
    std::uniform_real_distribution<> distrib(min_, max_);
    
    // Search non exixting keys
    constexpr size_t n_tests = (size_t) 1e6;
    for (size_t i = 0; i < n_tests; ++i) {
        data_t key;
        do {
            key = distrib(gen);
        } while (key == std::numeric_limits<data_t>::max()); // Do not search for the sentinel value

        // Make the query
        if constexpr (!approximate) {
        
            size_t search_res_lower = e.lower_bound_idx(key);
            size_t search_res_upper = e.upper_bound_idx(key);
            auto std_res_lower = std::lower_bound(v, v + sz, key);
            auto std_res_upper = std::upper_bound(v, v + sz, key);

            if (*std_res_lower != v[search_res_lower])
                throw std::runtime_error(test_name + " TEST FAIL: Lower - Non existing key test error, expected: " + std::to_string(std_res_lower - v) + " obtained: " + std::to_string(search_res_lower));

            if (*std_res_upper != v[search_res_upper])
                throw std::runtime_error(test_name + " TEST FAIL: Upper - Non existing key test error, expected: " + std::to_string(std_res_upper - v) + " obtained: " + std::to_string(search_res_upper));
        
        } else {
        
            // If approximate check only the lower bound
            auto search_res = e.search(key);
            size_t search_res_lower = std::distance(v, std::lower_bound(v + search_res.lo, v + search_res.hi, key));
            auto std_res_lower = std::lower_bound(v, v + sz, key);
            if (*std_res_lower != v[search_res_lower])
                throw std::runtime_error(test_name + " TEST FAIL: Lower - Non existing key test error, expected: " + std::to_string(std_res_lower - v) + " obtained: " + std::to_string(search_res_lower));
        
        }
    }
}

template <size_t sz>
void many_tests() {
    data_t * v = (data_t*)std::aligned_alloc(64, sizeof(data_t) * sz);

    std::mt19937 gen(seed);
    data_t min_ = std::numeric_limits<data_t>::lowest(), max_ = std::numeric_limits<data_t>::max();
    if (typeid(data_t) == typeid(double)) {
        // Reduce range for doubles, see why here: https://stackoverflow.com/a/66854452
        max_ /= 2;
        min_ = -max_;
    }
    std::uniform_real_distribution<> distrib(min_, max_);

    std::generate(v, v + sz, [&](){return distrib(gen); });
    ips4o::parallel::sort(v, v + sz);

    constexpr size_t log_sz = std::log2(sz);

    // EBS
    test<SIMD_Btree::ebs<data_t, log_sz>>(v, sz, "EBS" + std::to_string(sz));

    // B-Tree
    test<SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX2, 1>>(v, sz, "EBS line avx2 1 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX2, 2>>(v, sz, "EBS line avx2 2 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX2, 4>>(v, sz, "EBS line avx2 4 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX512, 1>>(v, sz, "EBS line avx512 1 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX512, 2>>(v, sz, "EBS line avx512 2 " + std::to_string(sz));
    test<SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX512, 4>>(v, sz, "EBS line avx512 4 " + std::to_string(sz));

    // Sampled B-Tree
    test<SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 1>>(v, sz, "B PLUS avx2 1 " + std::to_string(sz));
    test<SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 2>>(v, sz, "B PLUS avx2 2 " + std::to_string(sz));
    test<SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 4>>(v, sz, "B PLUS avx2 4 " + std::to_string(sz));
    test<SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 1>>(v, sz, "B PLUS avx512 1 " + std::to_string(sz));
    test<SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 2>>(v, sz, "B PLUS avx512 2 " + std::to_string(sz));
    test<SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 4>>(v, sz, "B PLUS avx512 4 " + std::to_string(sz));
    
    // B+ tree with leaves of different size
    test<SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 2, 4>>(v, sz, "B PLUS avx2 2-4 " + std::to_string(sz));
    test<SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 4, 1>>(v, sz, "B PLUS avx512 4-1 " + std::to_string(sz));

#if defined TEST_UINT32 || defined TEST_UINT64
    test<pgm::BTreePGMIndex<data_t>, true>(v, sz, "PGM Btree " + std::to_string(sz));
#endif

    std::free(v);
}

int main() {
    many_tests<2>();
    many_tests<7>();
    many_tests<8>();
    many_tests<20>();
    many_tests<63>();
    many_tests<64>();
    many_tests<65>();
    many_tests<1000>();
    many_tests<100'000>();
    many_tests<1'000'000>();
    
    std::cout << "Tests OK" << std::endl;
    return 0;
}
