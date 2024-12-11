#pragma once

#include <cmath>
#include <stdexcept>

#include <vector>

namespace searches {

    template <typename value_type>
    class ebs {
        void build_rec(const value_type* const left, const value_type* const right, const size_t pos) {
            const auto mid = left + (right - left) / 2;
            new (tree + pos - 1) value_type(*mid);
            if (2 * pos < tree_size) {
                build_rec(left, mid, 2 * pos);
                build_rec(mid, right, 2 * pos + 1);
            }
        }

    public:
        ~ebs() {
            if (tree != nullptr)
                std::free(tree);
        }

        void build(const value_type* const left, const value_type* const right) {
            tree_size = right - left;
            
            // Create a tmp copy of size (line_size+1)^k - 1 for some k
            std::vector<value_type> tmp_sequence(tree_size);
            std::copy(left, right, tmp_sequence.begin());
            
            log_tree_size = std::log2(tree_size) + 1;

            if (((tree_size + 1) & (tree_size)) != 0) {
                size_t padded_tree_size = std::pow(2, log_tree_size) - 1;
                tmp_sequence.resize(padded_tree_size);
                std::fill(tmp_sequence.begin() + tree_size, tmp_sequence.end(), std::numeric_limits<value_type>::max());
                //std::cout << "Padding added " << tree_size << " -> " << padded_tree_size << std::endl;
                tree_size = padded_tree_size;
            }

            if (tree != nullptr)
                std::free(tree);

            tree = (value_type *) std::aligned_alloc(64, sizeof(value_type) * tree_size);

            build_rec(tmp_sequence.data(), tmp_sequence.data() + tree_size, 1);
        }

        inline size_t search(const value_type& value) const {
            size_t b = 1;
            for (size_t l = 0; l < log_tree_size; ++l)
                b = 2 * b + std::less{}(tree[b - 1], value);

            return b - tree_size - 1;
        }

        inline size_t search(const value_type* const left, const value_type* const right, const value_type& value) const {
            return search(value);
        } 

        value_type * tree = nullptr;
        size_t tree_size;
        size_t log_tree_size;
    };
}
