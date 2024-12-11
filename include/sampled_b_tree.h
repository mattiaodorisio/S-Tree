#pragma once

#include "b_tree.h"

namespace SIMD_Btree {
    
    template <typename value_type,
              SIMD_ext ext = AVX512,
              size_t vectors_per_block = sizeof(value_type) / 4,
              size_t leaves_vectors_per_block = vectors_per_block>
    class b_plus_tree {
        using reg = RegisterSelector<value_type, ext>;
        static constexpr unsigned leaf_block_size = reg::vector_size * leaves_vectors_per_block;

    public:
        ~b_plus_tree() {
            clear();
        }

        void build(const value_type* const left, const value_type* const right) {
            // Input check
            if (std::is_sorted(left, right) == false)
                throw std::runtime_error("Cannot build btree from unsorted data");

            // Reset and deallocate the memory
            clear();

            // Copy the input vector
            leaves_number = right - left;
            leaf_nodes = (value_type *) std::aligned_alloc(64, sizeof(value_type) * (leaves_number + 1));
            std::copy(left, right, leaf_nodes);
            leaf_nodes[leaves_number] = std::numeric_limits<value_type>::max();

            if (leaves_number <= leaf_block_size) {
                return;
            }

            leaves_blocks = leaves_number / leaf_block_size;
            value_type * innner_nodes_tmp = (value_type *) std::aligned_alloc(64, sizeof(value_type) * leaves_blocks);

            for (size_t i = 0; i < leaves_blocks; ++i) {
                innner_nodes_tmp[i] = leaf_nodes[(i + 1) * leaf_block_size - 1];
            }

            inner_nodes.build(innner_nodes_tmp, innner_nodes_tmp + leaves_blocks);

            std::free(innner_nodes_tmp);
        }

        // Iterator interface
        template <typename RandomIt>
        void build(const RandomIt first, const RandomIt last) {
            build(static_cast<const value_type* const>(&(*first)), static_cast<const value_type* const>(&(*last)));
        }

        __attribute__((always_inline))
        inline size_t search(const value_type& value) const {
            
            using reg_t = reg::reg_type;
            reg_t x = reg::reg_set1(value);

            if (leaves_number <= leaf_block_size) [[ unlikely ]] {
                return block<value_type, ext, leaves_vectors_per_block>::block_rank(x, leaf_nodes);
            }

            size_t block_index = inner_nodes.search(value);
            return block_index * leaf_block_size + block<value_type, ext, leaves_vectors_per_block>::block_rank(x, leaf_nodes + (block_index * leaf_block_size));
        }

        size_t size_in_bytes() const {
            return sizeof(*this) + inner_nodes.size_in_bytes() + sizeof(value_type) * leaves_blocks;
        }

        void clear() {
            inner_nodes.clear();
            if (leaf_nodes != nullptr) {
                std::free(leaf_nodes);
            }
        }

    private:
        btree<value_type, ext, vectors_per_block> inner_nodes;
        size_t leaves_number = 0;
        size_t leaves_blocks = 0;
        value_type * leaf_nodes = nullptr;
    };
}
