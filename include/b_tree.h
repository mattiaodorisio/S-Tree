#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include <functional>

#pragma GCC target("avx2")
#pragma GCC target("avx512f")
#include <x86intrin.h>
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
    };

    // float avx2
    template <>
    class RegisterSelector<float, AVX2> {
        public:
        using reg_type = __m256;
        static constexpr size_t vector_size = 8;
        static constexpr auto reg_set1 = _mm256_set1_ps;
        static constexpr auto reg_load = _mm256_load_ps;
        static constexpr auto reg_cmp_ge_mask = std::bind(&_mm256_cmp_ps_mask, std::placeholders::_1, std::placeholders::_2, _CMP_GE_OQ);
    };

    // float avx512
    template <>
    class RegisterSelector<float, AVX512> {
        public:
        using reg_type = __m512;
        static constexpr size_t vector_size = 16;
        static constexpr auto reg_set1 = _mm512_set1_ps;
        static constexpr auto reg_load = _mm512_load_ps;
        static constexpr auto reg_cmp_ge_mask = std::bind(&_mm512_cmp_ps_mask, std::placeholders::_1, std::placeholders::_2, _CMP_GE_OQ);
    };

    // double avx2
    template <>
    class RegisterSelector<double, AVX2> {
        public:
        using reg_type = __m256d;
        static constexpr size_t vector_size = 4;
        static constexpr auto reg_set1 = _mm256_set1_pd;
        static constexpr auto reg_load = _mm256_load_pd;
        static constexpr auto reg_cmp_ge_mask = std::bind(&_mm256_cmp_pd_mask, std::placeholders::_1, std::placeholders::_2, _CMP_GE_OQ);
    };

    // double avx512
    template <>
    class RegisterSelector<double, AVX512> {
        public:
        using reg_type = __m512d;
        static constexpr size_t vector_size = 8;
        static constexpr auto reg_set1 = _mm512_set1_pd;
        static constexpr auto reg_load = _mm512_load_pd;
        static constexpr auto reg_cmp_ge_mask = std::bind(&_mm512_cmp_pd_mask, std::placeholders::_1, std::placeholders::_2, _CMP_GE_OQ);
    };


    // ********* tzcnt Selectors ********* //

    // The class tzcnt_selector is a wrapper to select the proper "trailing zero counter" function
    template <bool small>
    class tzcnt_selector {};

    template <>
    class tzcnt_selector<true> { public: static constexpr auto tzncnt_fun = __tzcnt_u32; };

    template <>
    class tzcnt_selector<false> { public: static constexpr auto tzncnt_fun = __tzcnt_u64; };


    // ********* Block ********* //

    // This class contains static operations on the block
    template <typename value_type, SIMD_ext ext, size_t vectors_per_block>
    class block {
    
        using reg = RegisterSelector<value_type, ext>;
        using reg_t = reg::reg_type;

    public:
        __attribute__((always_inline))
        static inline unsigned block_rank(const reg_t x, const value_type * y) {

            if constexpr (vectors_per_block != 1 && vectors_per_block != 2 && vectors_per_block != 4) {
                throw std::runtime_error("Unsupported number of vectors per block");
            }

            constexpr unsigned block_size = vectors_per_block * reg::vector_size;

            using mask_t = std::conditional<ext != AVX512 && block_size <= 32, __mmask8, __mmask16>::type;
            using mask_int_t = std::conditional<block_size <= 32, uint32_t, uint64_t>::type;
            constexpr auto tzcnt_fun = tzcnt_selector<block_size <= 32>::tzncnt_fun;
            constexpr bool bound_check_needed = block_size == 32 || block_size == 64;

            if constexpr (vectors_per_block == 1) {

                reg_t a = reg::reg_load(y);
                mask_t ca = reg::reg_cmp_ge_mask(a, x);

                if constexpr (bound_check_needed) {
                    return tzcnt_fun(ca);
                } else {
                    return tzcnt_fun((ca | (1 << (block_size))));
                }

            } else if constexpr (vectors_per_block == 2) {
                
                reg_t a = reg::reg_load(y);
                reg_t b = reg::reg_load(y + reg::vector_size);

                mask_t ca = reg::reg_cmp_ge_mask(a, x);
                mask_t cb = reg::reg_cmp_ge_mask(b, x);

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
            
                mask_t ca = reg::reg_cmp_ge_mask(a, x);
                mask_t cb = reg::reg_cmp_ge_mask(b, x);
                mask_t cc = reg::reg_cmp_ge_mask(c, x);
                mask_t cd = reg::reg_cmp_ge_mask(d, x);

                if constexpr (bound_check_needed) {
                    return tzcnt_fun((((mask_int_t)cd << reg::vector_size * 3) | ((mask_int_t)cc << reg::vector_size * 2) | (cb << reg::vector_size) | ca));
                } else { 
                    return tzcnt_fun((((mask_int_t)cd << reg::vector_size * 3) | ((mask_int_t)cc << reg::vector_size * 2) | (cb << reg::vector_size) | ca | (1 << block_size)));
                }
            }
        }
    };
    

    // ********* BTREE ********* //

    template <typename value_type, SIMD_ext ext = AVX512, size_t vectors_per_block = sizeof(value_type) / 4>
    class btree {

        using reg = RegisterSelector<value_type, ext>;
        using reg_t = reg::reg_type;
        static constexpr size_t line_size = reg::vector_size * vectors_per_block;

        void build_rec(const value_type* const left, const value_type* const right, const size_t pos, value_type * target_left, const size_t overall_size) const {
            auto mid_prev = left;
            for (size_t i = 0; i < line_size; ++i) {
                const auto mid = left + ((right - left + 1) / (line_size + 1)) * (i+1) - 1;
                new (target_left + (pos - 1) * line_size + i) value_type(*mid);
                const size_t new_pos = pos * (line_size + 1) + i - (line_size-1);
                if (new_pos < overall_size / line_size) {
                    build_rec(mid_prev, mid, new_pos, target_left, overall_size);
                }
                mid_prev = mid + 1;
            }
            const size_t new_pos = pos * (line_size + 1) + line_size - (line_size-1);
            if (new_pos < overall_size / line_size + 1) {
                build_rec(mid_prev, right, new_pos, target_left, overall_size);
            }
        }

        template <size_t iter_count>
        __attribute__((always_inline))
        inline size_t search_unrolled(const value_type& value) const {
            reg_t x = reg::reg_set1(value);
            size_t b = 1;
            if constexpr (iter_count == 0) { // deafult non-unrolled version
                for (size_t l = 0; l < log_tree_size - 1; ++l) {
                    size_t block_idx = (b-1) * line_size;
                    size_t cond_true = block<value_type, ext, vectors_per_block>::block_rank(x, tree + block_idx);
                    b = block_idx + b + cond_true + 1;
                }
            } else { // The number of iteration is known at compile time
                for (size_t l = 0; l < iter_count - 1; ++l) {
                    size_t block_idx = (b-1) * line_size;
                    size_t cond_true = block<value_type, ext, vectors_per_block>::block_rank(x, tree + block_idx);
                    b = block_idx + b + cond_true + 1;
                }
            }

            size_t block_idx = (b-1) * line_size;
            if (block_idx < tree_size) {
                size_t cond_true = block_idx < tree_size ? block<value_type, ext, vectors_per_block>::block_rank(x, tree + block_idx) : 0;
                b = block_idx + b + cond_true + 1;
                return b - tree_virtual_size / line_size - 1;
            } else {
                return (b - tree_reduced_size / line_size - 1) + (tree_size - tree_reduced_size);
            }
        }

        value_type * tree = nullptr;
        size_t tree_size = 0;
        size_t tree_virtual_size = 0;
        size_t tree_reduced_size = 0;
        size_t log_tree_size = 0;

    public:
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
            
            // The height virtual tree (the complete one, with dummy leaves)
            // tree_size == tree_virtual_size iff tree_size == (line_size+1)^k-1 for some k
            tree_virtual_size = std::pow(line_size + 1, log_tree_size) - 1;

            // The size of the tree without the last (potentially partial) level of leaves
            tree_reduced_size = std::pow(line_size + 1, log_tree_size - 1) - 1;

            // Add a padding to ensure that the tree has a size that is a multiple of the block size
            size_t padding_size = tree_size % line_size;
            padding_size = padding_size == 0 ? 0 : line_size - padding_size;

            tree = (value_type *) std::aligned_alloc(64, sizeof(value_type) * (tree_size + padding_size));
            for (size_t i = tree_size; i < tree_size + padding_size; ++i)
                tree[i] = std::numeric_limits<value_type>::max();

            value_type * tree_tmp = (value_type *) std::aligned_alloc(64, sizeof(value_type) * (tree_reduced_size));

            // Prepare the tree without the last level of leaves
            size_t last_level_size = tree_size - tree_reduced_size;
            size_t current_idx = 0, tree_tmp_idx = 0;
            while (last_level_size > 0) {
                if (current_idx % (line_size + 1) == line_size) {
                    tree_tmp[tree_tmp_idx++] = left[current_idx++];
                } else {
                    tree[tree_size - last_level_size] = left[current_idx++];
                    --last_level_size;
                }
            }
            std::copy(left + current_idx, right, tree_tmp + tree_tmp_idx);

            if (tree_reduced_size > 0) {
                build_rec(tree_tmp, tree_tmp + tree_reduced_size, 1, tree, tree_reduced_size);
            }

            std::free(tree_tmp);
        }

        // Iterator interface
        template <typename RandomIt>
        void build(const RandomIt first, const RandomIt last) {
            build(static_cast<const value_type* const>(&(*first)), static_cast<const value_type* const>(&(*last)));
        }

        __attribute__((always_inline))
        inline size_t search(const value_type& value) const {
            switch (log_tree_size) {
                case 1:  return search_unrolled<1ul>(value); break;
                case 2:  return search_unrolled<2ul>(value); break;
                case 3:  return search_unrolled<3ul>(value); break;
                case 4:  return search_unrolled<4ul>(value); break;
                case 5:  return search_unrolled<5ul>(value); break;
                case 6:  return search_unrolled<6ul>(value); break;
                case 7:  return search_unrolled<7ul>(value); break;
                case 8:  return search_unrolled<8ul>(value); break;
                case 9:  return search_unrolled<9ul>(value); break;
                case 10: return search_unrolled<10ul>(value); break;
                case 11: return search_unrolled<11ul>(value); break;
                case 12: return search_unrolled<12ul>(value); break;
                case 13: return search_unrolled<13ul>(value); break;
                case 14: return search_unrolled<14ul>(value); break;
                case 15: return search_unrolled<15ul>(value); break;
                default: return search_unrolled<0ul>(value); break;
            }
        }

        size_t size_in_bytes() const {
            return sizeof(*this);
        }

        void clear() {
            if (tree != nullptr) {
                std::free(tree);
                tree = nullptr;
                tree_size = 0;
            }
        }
    };
}
