#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>

#include "tree_utils.h"

namespace SIMD_Btree {

    template <typename value_type, size_t known_log_tree_size = 0>
    class ebs {
        void build_rec(const value_type* const left, const value_type* const right, const size_t pos, value_type * target_left, const size_t overall_size) const {
            const auto mid = left + (right - left) / 2;
            new (target_left + pos - 1) value_type(*mid);
            if (2 * pos < overall_size) {
                build_rec(left, mid, 2 * pos, target_left, overall_size);
                build_rec(mid, right, 2 * pos + 1, target_left, overall_size);
            }
        }

    public:
        ebs() = default;

        ebs(const value_type* const left, const value_type* const right) {
            build(left, right);
        }

        ~ebs() {
            if (tree != nullptr)
                std::free(tree);
        }

        void build(const value_type* const left, const value_type* const right) {
            // Input check
            if (std::is_sorted(left, right) == false)
                throw std::runtime_error("Cannot build ebs layout from unsorted data");

            tree_size = right - left;
            log_tree_size = std::log2(tree_size);
            size_t tree_reduced_size = (1ul << log_tree_size) - 1;

            if constexpr (known_log_tree_size != 0) {
                if (known_log_tree_size != log_tree_size) {
                    throw std::runtime_error("Tree size not properly templated");
                }
            }
            
            std::vector<value_type> tmp_sequence(tree_reduced_size);

            if (tree != nullptr)
                std::free(tree);

            // Allocate the memory, enable huge page
            constexpr int P = 1 << 21;
            tree = (value_type *) std::aligned_alloc(P, sizeof(value_type) * tree_size + 1);
            madvise(tree, sizeof(value_type) * tree_size + 1, MADV_HUGEPAGE);
            
            tree[tree_size] = std::numeric_limits<value_type>::max();

            // Prepare the tree without the last level of leaves
            size_t last_level_size = tree_size - tree_reduced_size;
            size_t current_idx = 0, tree_tmp_idx = 0;
            while (last_level_size > 0) {
                if (current_idx % 2 == 1) {
                    tmp_sequence[tree_tmp_idx++] = left[current_idx++];
                } else {
                    tree[tree_size - last_level_size] = left[current_idx++];
                    --last_level_size;
                }
            }
            std::copy(left + current_idx, right, tmp_sequence.data() + tree_tmp_idx);

            build_rec(tmp_sequence.data(), tmp_sequence.data() + tree_reduced_size, 1, tree, tree_reduced_size);
        }

        template <size_t iter_count, typename comp>
        FORCE_INLINE size_t search_unrolled(const value_type& value) const {

            size_t b = 1;
            for (size_t l = 0; l < iter_count; ++l)
                b = 2 * b + comp{}(tree[b - 1], value);

            value_type t[1];
            t[0] = std::numeric_limits<value_type>::max();

            bool cond = b <= tree_size;
            const size_t b_half = b;
            value_type *loc = cond ? tree + b - 1 : t;
            b = 2 * b + comp{}(*loc, value);

            // This is correct:
            //size_t result = cond ? b - std::pow(2, log_tree_size + 1) : b / 2 - std::pow(2, log_tree_size) + tree_size - (std::pow(2, log_tree_size) - 1);
            
            // Also this is correct:
            //size_t result = cond ? b - std::pow(2, log_tree_size + 1) : b / 2 - std::pow(2, log_tree_size + 1) + tree_size + 1;
            
            // Thus this:
            //size_t result = cond ? b : b / 2 + tree_size + 1;
            //result -= std::pow(2, log_tree_size + 1);
            
            constexpr size_t tree_virtual_size = 1ul << (iter_count + 1);
            size_t result = b - tree_virtual_size;
            result = cond ? result : result - b_half + tree_size + 1;

            return result;
        }

        template <typename comp = std::less<value_type>>
        FORCE_INLINE size_t search(const value_type& value) const {
            if constexpr (known_log_tree_size != 0) {
                return search_unrolled<known_log_tree_size, comp>(value);
            } else {
                switch (log_tree_size) {
                case 1: return search_unrolled<1ul, comp>(value); break;
                case 2: return search_unrolled<2ul, comp>(value); break;
                case 3: return search_unrolled<3ul, comp>(value); break;
                case 4: return search_unrolled<4ul, comp>(value); break;
                case 5: return search_unrolled<5ul, comp>(value); break;
                case 6: return search_unrolled<6ul, comp>(value); break;
                case 7: return search_unrolled<7ul, comp>(value); break;
                case 8: return search_unrolled<8ul, comp>(value); break;
                case 9: return search_unrolled<9ul, comp>(value); break;
                case 10: return search_unrolled<10ul, comp>(value); break;
                case 11: return search_unrolled<11ul, comp>(value); break;
                case 12: return search_unrolled<12ul, comp>(value); break;
                case 13: return search_unrolled<13ul, comp>(value); break;
                case 14: return search_unrolled<14ul, comp>(value); break;
                case 15: return search_unrolled<15ul, comp>(value); break;
                case 16: return search_unrolled<16ul, comp>(value); break;
                case 17: return search_unrolled<17ul, comp>(value); break;
                case 18: return search_unrolled<18ul, comp>(value); break;
                case 19: return search_unrolled<19ul, comp>(value); break;
                case 20: return search_unrolled<20ul, comp>(value); break;
                case 21: return search_unrolled<21ul, comp>(value); break;
                case 22: return search_unrolled<22ul, comp>(value); break;
                case 23: return search_unrolled<23ul, comp>(value); break;
                case 24: return search_unrolled<24ul, comp>(value); break;
                case 25: return search_unrolled<25ul, comp>(value); break;
                case 26: return search_unrolled<26ul, comp>(value); break;
                case 27: return search_unrolled<27ul, comp>(value); break;
                case 28: return search_unrolled<28ul, comp>(value); break;
                case 29: return search_unrolled<29ul, comp>(value); break;
                case 30: return search_unrolled<30ul, comp>(value); break;
                case 31: return search_unrolled<31ul, comp>(value); break;
                case 32: return search_unrolled<32ul, comp>(value); break;
                case 33: return search_unrolled<33ul, comp>(value); break;
                case 34: return search_unrolled<34ul, comp>(value); break;
                case 35: return search_unrolled<35ul, comp>(value); break;
                case 36: return search_unrolled<36ul, comp>(value); break;
                case 37: return search_unrolled<37ul, comp>(value); break;
                case 38: return search_unrolled<38ul, comp>(value); break;
                case 39: return search_unrolled<39ul, comp>(value); break;
                case 40: return search_unrolled<40ul, comp>(value); break;
                case 41: return search_unrolled<41ul, comp>(value); break;
                case 42: return search_unrolled<42ul, comp>(value); break;
                case 43: return search_unrolled<43ul, comp>(value); break;
                case 44: return search_unrolled<44ul, comp>(value); break;
                case 45: return search_unrolled<45ul, comp>(value); break;
                case 46: return search_unrolled<46ul, comp>(value); break;
                case 47: return search_unrolled<47ul, comp>(value); break;
                case 48: return search_unrolled<48ul, comp>(value); break;
                case 49: return search_unrolled<49ul, comp>(value); break;
                case 50: return search_unrolled<50ul, comp>(value); break;
                case 51: return search_unrolled<51ul, comp>(value); break;
                case 52: return search_unrolled<52ul, comp>(value); break;
                case 53: return search_unrolled<53ul, comp>(value); break;
                case 54: return search_unrolled<54ul, comp>(value); break;
                case 55: return search_unrolled<55ul, comp>(value); break;
                case 56: return search_unrolled<56ul, comp>(value); break;
                case 57: return search_unrolled<57ul, comp>(value); break;
                case 58: return search_unrolled<58ul, comp>(value); break;
                case 59: return search_unrolled<59ul, comp>(value); break;
                case 60: return search_unrolled<60ul, comp>(value); break;
                case 61: return search_unrolled<61ul, comp>(value); break;
                case 62: return search_unrolled<62ul, comp>(value); break;
                default: throw std::runtime_error("Unsupported size error"); break;
                }
            }
        }

        inline size_t lower_bound_idx(const value_type& value) const {
            return search<std::less<value_type>>(value);
        }

        inline size_t upper_bound_idx(const value_type& value) const {
            return search<std::less_equal<value_type>>(value);
        }

        inline size_t size() const {
            return tree_size;
        }

        // Support operator[]
        static inline constexpr bool support_access() {
            return true;
        }

        value_type operator[](const size_t idx) const {
            // This function converts the index in the sorted array in the of the eytzinger layout

            // Compute how many leaves are in the last (possibly) non-complete level
            const size_t exceeding_tree = tree_size - (1 << log_tree_size) + 1;
            
            // Compute the number of nodes of the largest subtree that contains all exceeding_tree leaves but none of the other leaves
            const size_t ok_indexes = 2 * exceeding_tree;
            
            // Compute the actual index to look for depending on ok_indexes
            const size_t adjusted_idx = idx < ok_indexes ? idx : idx - exceeding_tree;
            
            // OBSERVATION 1: Now forget about the adjustment and treat the tree as complete (small fix on the height later OBS. 3)
            // OBSERVATION 2: The level of the node (distance from the leaves) is equal to the number of trailing zeros in adjusted_idx
            /* Example:
             *       4                         100                         2 trailing zeros -> level 2 -> depth 0
             *    2     6        ->       010       110                    1 trailing zero  -> level 1 -> depth 1
             *  1   3 5   7            001  011   100  111                 0 trailing zero  -> level 0 -> depth 2
             */
            const size_t trailing_zeros = __builtin_ctzl(adjusted_idx + 1);
            size_t depth = log_tree_size - trailing_zeros;

            // OBSERVATION 3: If the index has been adjusted 'use' the tree without the partial leaves
            depth = idx < ok_indexes ? depth : depth - 1;
            
            // In the Eytzinger layout the nodes at depth 'depth' start at index 2^depth - 1
            const size_t level_start = (1 << depth) - 1;

            // OBSERVATION 4: Given a level, the index of a node at that level is obtained removing the trailing zeros and the least significant one  
            const size_t offset = ((adjusted_idx + 1) >> (trailing_zeros + 1));

            // Get the new index and return the element
            const size_t new_idx = level_start + offset;
            return tree[new_idx];
        }

        void clear() {
            if (tree != nullptr) {
                std::free(tree);
                tree = nullptr;
                tree_size = 0;
                log_tree_size = 0;
            }
        }

        value_type * tree = nullptr;
        size_t tree_size;
        size_t log_tree_size;
    };
}
