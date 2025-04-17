#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>

#include "tree_utils.h"

#pragma GCC target("avx2")
#pragma GCC target("avx512f")
#include <x86intrin.h>
#include <immintrin.h>
#include <bits/stdc++.h>

namespace SIMD_Btree {

    enum SIMD_ext {
        AVX2,
        AVX512
    };

    // ********* Register Selector ********* //

    // The class RegisterSelector is specialized for each type.
    // It abstracts the intrinsics and the register types from the btree class.
    // Everything is done at compile time

    template <typename T, SIMD_ext ext>
    class RegisterSelector;

    // int32 avx2
    template <>
    class RegisterSelector<int32_t, AVX2> { 
        public:
        using reg_type = __m256i;
        static constexpr size_t vector_size = 8;
        static constexpr auto reg_set1 = _mm256_set1_epi32;
        static constexpr auto reg_load = _mm256_load_epi32;
        static constexpr auto reg_cmp_ge_mask = _mm256_cmpge_epi32_mask;
        static constexpr auto reg_cmp_gt_mask = _mm256_cmpgt_epi32_mask;
    };

    // int32 avx512
    template <>
    class RegisterSelector<int32_t, AVX512> { 
        public:
        using reg_type = __m512i;
        static constexpr size_t vector_size = 16;
        static constexpr auto reg_set1 = _mm512_set1_epi32;
        static constexpr auto reg_load = _mm512_load_epi32;
        static constexpr auto reg_cmp_ge_mask = _mm512_cmpge_epi32_mask;
        static constexpr auto reg_cmp_gt_mask = _mm512_cmpgt_epi32_mask;
    };

    // uint32 avx2
    template <>
    class RegisterSelector<uint32_t, AVX2> {
        public:
        using reg_type = __m256i;
        static constexpr size_t vector_size = 8;
        static constexpr auto reg_set1 = _mm256_set1_epi32;
        static constexpr auto reg_load = _mm256_load_epi32;
        static constexpr auto reg_cmp_ge_mask = _mm256_cmpge_epu32_mask;
        static constexpr auto reg_cmp_gt_mask = _mm256_cmpgt_epu32_mask;
    };

    // uint32 avx512
    template <>
    class RegisterSelector<uint32_t, AVX512> {
        public:
        using reg_type = __m512i;
        static constexpr size_t vector_size = 16;
        static constexpr auto reg_set1 = _mm512_set1_epi32;
        static constexpr auto reg_load = _mm512_load_epi32;
        static constexpr auto reg_cmp_ge_mask = _mm512_cmpge_epu32_mask;
        static constexpr auto reg_cmp_gt_mask = _mm512_cmpgt_epu32_mask;
    };

    // int64 avx2
    template <>
    class RegisterSelector<int64_t, AVX2> {
        public:
        using reg_type = __m256i;
        static constexpr size_t vector_size = 4;
        static constexpr auto reg_set1 = _mm256_set1_epi64x;
        static constexpr auto reg_load = _mm256_load_epi64;
        static constexpr auto reg_cmp_ge_mask = _mm256_cmpge_epi64_mask;
        static constexpr auto reg_cmp_gt_mask = _mm256_cmpgt_epi64_mask;
    };

    // int64 avx512
    template <>
    class RegisterSelector<int64_t, AVX512> {
        public:
        using reg_type = __m512i;
        static constexpr size_t vector_size = 8;
        static constexpr auto reg_set1 = _mm512_set1_epi64;
        static constexpr auto reg_load = _mm512_load_epi64;
        static constexpr auto reg_cmp_ge_mask = _mm512_cmpge_epi64_mask;
        static constexpr auto reg_cmp_gt_mask = _mm512_cmpgt_epi64_mask;
    };
    
    // uint64 avx2
    template <>
    class RegisterSelector<uint64_t, AVX2> {
        public:
        using reg_type = __m256i;
        static constexpr size_t vector_size = 4;
        static constexpr auto reg_set1 = _mm256_set1_epi64x;
        static constexpr auto reg_load = _mm256_load_epi64;
        static constexpr auto reg_cmp_ge_mask = _mm256_cmpge_epu64_mask;
        static constexpr auto reg_cmp_gt_mask = _mm256_cmpgt_epu64_mask;
    };

    // uint64 avx512
    template <>
    class RegisterSelector<uint64_t, AVX512> {
        public:
        using reg_type = __m512i;
        static constexpr size_t vector_size = 8;
        static constexpr auto reg_set1 = _mm512_set1_epi64;
        static constexpr auto reg_load = _mm512_load_epi64;
        static constexpr auto reg_cmp_ge_mask = _mm512_cmpge_epu64_mask;
        static constexpr auto reg_cmp_gt_mask = _mm512_cmpgt_epu64_mask;
    };

    // float avx2
    template <>
    class RegisterSelector<float, AVX2> {
        public:
        using reg_type = __m256;
        static constexpr size_t vector_size = 8;
        static constexpr auto reg_set1 = _mm256_set1_ps;
        static constexpr auto reg_load = _mm256_load_ps;
        
        static FORCE_INLINE auto reg_cmp_ge_mask(const reg_type a, const reg_type b) {
            return _mm256_cmp_ps_mask(a, b, _CMP_GE_OQ);
        }

        static FORCE_INLINE auto reg_cmp_gt_mask(const reg_type a, const reg_type b) {
            return _mm256_cmp_ps_mask(a, b, _CMP_GT_OQ);
        }
    };

    // float avx512
    template <>
    class RegisterSelector<float, AVX512> {
        public:
        using reg_type = __m512;
        static constexpr size_t vector_size = 16;
        static constexpr auto reg_set1 = _mm512_set1_ps;
        static constexpr auto reg_load = _mm512_load_ps;

        static FORCE_INLINE auto reg_cmp_ge_mask(const reg_type a, const reg_type b) {
            return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
        }

        static FORCE_INLINE auto reg_cmp_gt_mask(const reg_type a, const reg_type b) {
            return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
        }
    };

    // double avx2
    template <>
    class RegisterSelector<double, AVX2> {
        public:
        using reg_type = __m256d;
        static constexpr size_t vector_size = 4;
        static constexpr auto reg_set1 = _mm256_set1_pd;
        static constexpr auto reg_load = _mm256_load_pd;

        static FORCE_INLINE auto reg_cmp_ge_mask(const reg_type a, const reg_type b) {
            return _mm256_cmp_pd_mask(a, b, _CMP_GE_OQ);
        }

        static FORCE_INLINE auto reg_cmp_gt_mask(const reg_type a, const reg_type b) {
            return _mm256_cmp_pd_mask(a, b, _CMP_GT_OQ);
        }
    };

    // double avx512
    template <>
    class RegisterSelector<double, AVX512> {
        public:
        using reg_type = __m512d;
        static constexpr size_t vector_size = 8;
        static constexpr auto reg_set1 = _mm512_set1_pd;
        static constexpr auto reg_load = _mm512_load_pd;

        static FORCE_INLINE auto reg_cmp_ge_mask(const reg_type a, const reg_type b) {
            return _mm512_cmp_pd_mask(a, b, _CMP_GE_OQ);
        }

        static FORCE_INLINE auto reg_cmp_gt_mask(const reg_type a, const reg_type b) {
            return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
        }

    };

    // ********* tzcnt Selectors ********* //

#if false
    // The class tzcnt_selector is a wrapper to select the proper "trailing zero counter" function
    template <bool small>
    class tzcnt_selector {};

    template <>
    class tzcnt_selector<true> { public: static constexpr auto tzncnt_fun = __tzcnt_u32; };

    template <>
    class tzcnt_selector<false> { public: static constexpr auto tzncnt_fun = __tzcnt_u64; };
#endif

    static FORCE_INLINE int ctz (unsigned int x) { return __builtin_ctz(x); };
    static FORCE_INLINE int ctzl (unsigned long x) { return __builtin_ctzl(x); };

    // The class tzcnt_selector is a wrapper to select the proper "trailing zero counter" function
    template <bool small>
    class tzcnt_selector {};

    template <>
    class tzcnt_selector<true> { public: static constexpr auto tzncnt_fun = ctz; };

    template <>
    class tzcnt_selector<false> { public: static constexpr auto tzncnt_fun = ctzl; };

    // ********* Block ********* //

    // This class contains static operations on the block
    template <typename value_type, SIMD_ext ext, size_t vectors_per_block>
    class block {
    
        using reg = RegisterSelector<value_type, ext>;
        using reg_t = reg::reg_type;

    public:
        template <bool lower_bound>
        static FORCE_INLINE unsigned block_rank(const reg_t x, const value_type * y) {

            if constexpr (vectors_per_block != 1 && vectors_per_block != 2 && vectors_per_block != 4) {
                throw std::runtime_error("Unsupported number of vectors per block");
            }

            constexpr unsigned block_size = vectors_per_block * reg::vector_size;

            using mask_t = std::conditional<ext != AVX512 && block_size <= 32, __mmask8, __mmask16>::type;
            using mask_int_t = std::conditional<block_size <= 32, uint32_t, uint64_t>::type;
            constexpr auto tzcnt_fun = tzcnt_selector<block_size <= 32>::tzncnt_fun;
            constexpr bool bound_check_needed = block_size == 32 || block_size == 64;
            constexpr auto comp_fun = lower_bound ? reg::reg_cmp_ge_mask : reg::reg_cmp_gt_mask;

            if constexpr (vectors_per_block == 1) {

                reg_t a = reg::reg_load(y);
                mask_t ca = comp_fun(a, x);

                if constexpr (bound_check_needed) {
                    return tzcnt_fun(ca);
                } else {
                    return tzcnt_fun((ca | (1 << (block_size))));
                }

            } else if constexpr (vectors_per_block == 2) {
                
                reg_t a = reg::reg_load(y);
                reg_t b = reg::reg_load(y + reg::vector_size);

                mask_t ca = comp_fun(a, x);
                mask_t cb = comp_fun(b, x);

                if constexpr (bound_check_needed) {
                    return tzcnt_fun(((cb << reg::vector_size) | ca ));
                } else {
                    return tzcnt_fun(((cb << reg::vector_size) | ca | (1 << block_size)));
                }

            } else if constexpr (vectors_per_block == 4) {

                reg_t a = reg::reg_load(y);
                reg_t b = reg::reg_load(y + reg::vector_size);
                reg_t c = reg::reg_load(y + 2 * reg::vector_size);
                reg_t d = reg::reg_load(y + 3 * reg::vector_size);
            
                mask_t ca = comp_fun(a, x);
                mask_t cb = comp_fun(b, x);
                mask_t cc = comp_fun(c, x);
                mask_t cd = comp_fun(d, x);

                if constexpr (bound_check_needed) {
                    return tzcnt_fun((((mask_int_t)cd << reg::vector_size * 3) | ((mask_int_t)cc << reg::vector_size * 2) | (cb << reg::vector_size) | ca));
                } else { 
                    return tzcnt_fun((((mask_int_t)cd << reg::vector_size * 3) | ((mask_int_t)cc << reg::vector_size * 2) | (cb << reg::vector_size) | ca | (1 << block_size)));
                }
            }
        }
    };
    

    // ********* BTREE ********* //

    template <typename value_type, size_t known_log_tree_size = 0, SIMD_ext ext = AVX512, size_t vectors_per_block = 1>
    class btree {

        using reg = RegisterSelector<value_type, ext>;
        using reg_t = reg::reg_type;
        static constexpr size_t line_size = reg::vector_size * vectors_per_block;
        constexpr static bool is_power_of_two(const size_t n) { return (n & (n - 1)) == 0; }
        static_assert(is_power_of_two(line_size));

        // Precomputed (at compile time) table of powers of the tree fanout (line_size + 1)
        static consteval auto fanout_powers() {
            std::array<size_t, 16> arr;
            arr[0] = 1;
            for(size_t i = 1; i < arr.size(); ++i) {
                arr[i] = arr[i-1] * (line_size + 1);
            }
            return arr;
        }

        static consteval size_t const_fanout_pow(const size_t exponent) {
            return (fanout_powers()[exponent]);
        }

        static size_t fanout_pow(const size_t exponent) {
            return (fanout_powers()[exponent]);
        }

        void build_bottom_up_step(value_type * from_left, value_type * from_right, value_type * to_left) {
            const size_t current_size = from_right - from_left;
            const size_t current_internal_size = current_size / (line_size + 1);

            value_type * from_reading_right = from_right;
            value_type * to_writing_right = to_left + current_internal_size;
            while (from_reading_right > from_left) {
                for (size_t j = 0; j < line_size; ++j) {
                    *(--from_right) = (*(--from_reading_right));
                }
                if (from_reading_right > from_left) {
                    *(--to_writing_right) = (*(--from_reading_right));
                }
            }
        }

        void build_bottom_up(value_type * left, value_type * right) {
            size_t current_size = right - left;
            size_t current_internal_size = current_size / (line_size + 1);

            if (current_internal_size == 0) {
                return;
            }

            value_type * buffer = new value_type[current_internal_size];

            while (current_internal_size > 0) {
                build_bottom_up_step(left, left + current_size, buffer);
                std::copy(buffer, buffer + current_internal_size, left);
                current_size = current_internal_size;
                current_internal_size = current_size / (line_size + 1);
            }

            delete[] buffer;
        }

        template <size_t iter_count, bool lower_bound>
        FORCE_INLINE size_t search_unrolled(const value_type& value) const {
            reg_t x = reg::reg_set1(value);
            size_t b = 1;
            if constexpr (iter_count == 0) { // deafult non-unrolled version
                for (size_t l = 0; l < log_tree_size - 1; ++l) {
                    size_t block_idx = (b-1) * line_size;
                    size_t cond_true = block<value_type, ext, vectors_per_block>::template block_rank<lower_bound>(x, tree + block_idx);
                    b = block_idx + b + cond_true + 1;
                }
            } else { // The number of iteration is known at compile time
                for (size_t l = 0; l < iter_count - 1; ++l) {
                    size_t block_idx = (b-1) * line_size;
                    size_t cond_true = block<value_type, ext, vectors_per_block>::template block_rank<lower_bound>(x, tree + block_idx);
                    b = block_idx + b + cond_true + 1;
                }
            }

            if constexpr(iter_count > 0) { // Some of the constants can be precomputed
                size_t block_idx = (b-1) * line_size;
                if (block_idx < tree_size) {
                    size_t cond_true = block<value_type, ext, vectors_per_block>::template block_rank<lower_bound>(x, tree + block_idx);
                    b = block_idx + b + cond_true + 1;
                    return b - (const_fanout_pow(iter_count) - 1) / line_size - 1;
                } else {
                    return b - (const_fanout_pow(iter_count - 1) - 1) / line_size - 1 + tree_exceeding_leaves;
                }
            } else {
                size_t block_idx = (b-1) * line_size;
                if (block_idx < tree_size) {
                    size_t cond_true = block<value_type, ext, vectors_per_block>::template block_rank<lower_bound>(x, tree + block_idx);
                    b = block_idx + b + cond_true + 1;
                    return b - tree_virtual_size / line_size - 1;
                } else {
                    return (b - (tree_size - tree_exceeding_leaves) / line_size - 1) + tree_exceeding_leaves;
                }
            }
        }

        value_type * tree = nullptr;
        size_t tree_size = 0;
        size_t tree_virtual_size = 0;
        size_t log_tree_size = 0;
        size_t tree_exceeding_nodes;
        size_t tree_exceeding_leaves;

    public:
    
        static constexpr size_t log_size(const size_t n) {
            return std::log2(n) / std::log2(line_size + 1) + 1;
        }

        btree() = default;

        btree(const value_type* const left, const value_type* const right) {
            build(left, right);
        }

        ~btree() {
            clear();
        }

        void build(const value_type* const left, const value_type* const right) {
            // Input check
            if (std::is_sorted(left, right) == false)
                throw std::runtime_error("Cannot build btree from unsorted data");

            // Reset and deallocate the memory
            clear();

            // The size of the data
            tree_size = right - left;

            // The height of the tree
            log_tree_size = std::log2(tree_size) / std::log2(line_size + 1) + 1;

            if constexpr (known_log_tree_size != 0) {
                if (known_log_tree_size != log_tree_size) {
                    throw std::runtime_error("Tree size not properly templated");
                }
            }
            
            // The height virtual tree (the complete one, with dummy leaves)
            // tree_size == tree_virtual_size iff tree_size == (line_size+1)^k-1 for some k
            tree_virtual_size = fanout_pow(log_tree_size) - 1;

            // The size of the tree without the last (potentially partial) level of leaves
            size_t tree_reduced_size = fanout_pow(log_tree_size - 1) - 1;
            tree_exceeding_leaves = tree_size - tree_reduced_size;

            // Add a padding to ensure that the tree has a size that is a multiple of the block size
            size_t padding_size = tree_size % line_size;
            padding_size = padding_size == 0 ? 0 : line_size - padding_size;

            // Allocate the memory, enable huge page
            constexpr int P = 1 << 21;
            tree = (value_type *) std::aligned_alloc(P, sizeof(value_type) * (tree_size + padding_size));
            madvise(tree, sizeof(value_type) * (tree_size + padding_size), MADV_HUGEPAGE);

            for (size_t i = tree_size; i < tree_size + padding_size; ++i)
                tree[i] = std::numeric_limits<value_type>::max();

            // Prepare the tree without the last level of leaves
            size_t last_level_size = tree_exceeding_leaves;
            size_t current_idx = 0, tree_tmp_idx = 0;
            while (last_level_size > 0) {
                if (current_idx % (line_size + 1) == line_size) {
                    tree[tree_tmp_idx++] = left[current_idx++];
                } else {
                    tree[tree_size - last_level_size] = left[current_idx++];
                    --last_level_size;
                }
            }
            tree_exceeding_nodes = current_idx;
            std::copy(left + current_idx, right, tree + tree_tmp_idx);

            if (tree_reduced_size > 0) {
                build_bottom_up(tree, tree + tree_reduced_size);
            }
        }

        // Iterator interface
        template <typename RandomIt>
        void build(const RandomIt first, const RandomIt last) {
            build(static_cast<const value_type* const>(&(*first)), static_cast<const value_type* const>(&(*last)));
        }

        template <bool lower_bound>
        FORCE_INLINE size_t search(const value_type& value) const {
            if constexpr (known_log_tree_size != 0) {
                return search_unrolled<known_log_tree_size, lower_bound>(value);
            } else {
                switch (log_tree_size) {
                    case 1:  return search_unrolled<1ul, lower_bound>(value); break;
                    case 2:  return search_unrolled<2ul, lower_bound>(value); break;
                    case 3:  return search_unrolled<3ul, lower_bound>(value); break;
                    case 4:  return search_unrolled<4ul, lower_bound>(value); break;
                    case 5:  return search_unrolled<5ul, lower_bound>(value); break;
                    case 6:  return search_unrolled<6ul, lower_bound>(value); break;
                    case 7:  return search_unrolled<7ul, lower_bound>(value); break;
                    case 8:  return search_unrolled<8ul, lower_bound>(value); break;
                    case 9:  return search_unrolled<9ul, lower_bound>(value); break;
                    case 10: return search_unrolled<10ul, lower_bound>(value); break;
                    case 11: return search_unrolled<11ul, lower_bound>(value); break;
                    case 12: return search_unrolled<12ul, lower_bound>(value); break;
                    case 13: return search_unrolled<13ul, lower_bound>(value); break;
                    case 14: return search_unrolled<14ul, lower_bound>(value); break;
                    case 15: return search_unrolled<15ul, lower_bound>(value); break;
                    default: return search_unrolled<0ul, lower_bound>(value); break;
                }
            }
        }

        inline size_t lower_bound_idx(const value_type& value) const {
            return search<true>(value);
        }

        inline size_t upper_bound_idx(const value_type& value) const {
            return search<false>(value);
        }

        // Support operator[]
        static inline constexpr bool support_access() {
            return true;
        }

        value_type operator[](const size_t idx) const {
            // Compute the actual index to look for depending on the exceeding nodes and leaves
            size_t adjusted_idx = idx < tree_exceeding_nodes ? idx : idx - tree_exceeding_leaves;
            ++adjusted_idx;

            // Can this be improved (like in the binary EBS) by looking at the index binary representation?
            size_t fanout_power = line_size + 1;
            size_t depth = log_tree_size - 1;

            while (adjusted_idx % fanout_power == 0) {
                fanout_power *= line_size + 1;
                --depth;
            }

            // Offset = smaller nodes on this level - smaller nodes that are on the upper levels
            /* Example (with line_size = 2 and fanout = 3):
             *                           9              ...
             *        3         6                   12  ...
             *  1  2     4   5     7  8     10  11      ...
             * The offset of 10 is given by its rank on the current level 10, minus the nodes at a higher levels 3 (3, 6, and 9)
             */
            size_t offset = adjusted_idx / (fanout_power / (line_size + 1)) - adjusted_idx / fanout_power - 1;
            const size_t level_start = fanout_pow(idx < tree_exceeding_nodes ? depth : depth - 1) - 1;
            return tree[level_start + offset];
        }

        size_t size_in_bytes() const {
            return sizeof(*this);
        }

        size_t size_all_bytes() const {
            return sizeof(*this) + tree_size * sizeof(value_type);
        }

        inline size_t height() const {
            return log_tree_size;
        }

        inline size_t size() const {
            return tree_size;
        }

        void clear() {
            if (tree != nullptr) {
                std::free(tree);
                tree = nullptr;
                tree_size = 0;
                log_tree_size = 0;
            }
        }
    };
}
