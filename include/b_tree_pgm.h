/**
 * This is a work in progress.
 * The idea is to use a one-layer PGM and replace the recursive search with a BTree.
 */
#pragma once

#include "piecewise_linear_model.hpp"
#include "pgm_index.hpp"
#include "b_tree.h"

#include <vector>

namespace pgm {
    /**
     * A variant of @ref PGMIndex that builds a top-level btree to speed up the search on the segments.
     *
     * @tparam K the type of the indexed keys
     * @tparam Epsilon controls the size of the returned search range
     * @tparam Floating the floating-point type to use for slopes
     */
    template<typename K, size_t Epsilon = 64, typename Floating = float>
    class BTreePGMIndex {
    protected:
        static_assert(Epsilon > 0);

        using Segment = typename PGMIndex<K, Epsilon, 0, Floating>::Segment;

        struct SegmentData {
            Floating slope;    ///< The slope of the segment.
            int32_t intercept; ///< The intercept of the segment.

            SegmentData() = default;

            SegmentData(Segment &s) : slope(s.slope), intercept(s.intercept) {}

            inline size_t operator()(const K &origin, const K &k) const {
                auto pos = int64_t(slope * (k - origin)) + intercept;
                return pos > 0 ? size_t(pos) : 0ull;
            }
        };

        size_t n;                           ///< The number of elements this index was built on.
        K first_key;                        ///< The smallest segment key.
        std::vector<SegmentData> segments;  ///< The segments composing the index.
        SIMD_Btree::btree<K> first_keys;    ///< The btree on the first keys
        
    public:

        static constexpr size_t epsilon_value = Epsilon;

        /**
         * Constructs an empty index.
         */
        BTreePGMIndex() = default;

        /**
         * Constructs the index on the given sorted vector.
         * @param data the vector of keys, must be sorted
         */
        explicit BTreePGMIndex(const std::vector<K> &data) : BTreePGMIndex(data.begin(), data.end()) {}

        /**
         * Constructs the index on the sorted keys in the range [first, last).
         * @param first, last the range containing the sorted keys to be indexed
         */
        template<typename RandomIt>
        BTreePGMIndex(RandomIt first, RandomIt last) {
            build(first, last); 
        }

        template<typename RandomIt>
        void build(const RandomIt first, const RandomIt last) {
            n = std::distance(first, last);
            first_key = n ? *first : K(0);
            segments.clear();
            first_keys.clear();

            if (n == 0)
                return;

            std::vector<Segment> tmp;
            std::vector<K> tmp_keys;
            std::vector<size_t> offsets;
            PGMIndex<K, Epsilon, 0, Floating>::build(first, last, Epsilon, 0, tmp, offsets);

            segments.reserve(tmp.size());
            tmp_keys.reserve(tmp.size());
            for (auto &x: tmp) {
                segments.push_back(x);
                tmp_keys.push_back(x.key);
            }
            
            first_keys.build(tmp_keys.begin(), tmp_keys.end());
        }

        /**
         * Returns the approximate position and the range where @p key can be found.
         * @param key the value of the element to search for
         * @return a struct with the approximate position and bounds of the range
         */
        ApproxPos search(const K &key) const {
            auto k = std::max(first_key, key);
            size_t seg_idx = first_keys.upper_bound_idx(k) - 1;
            auto pos = std::min<size_t>(segments[seg_idx](first_keys[seg_idx], k), segments[seg_idx+1].intercept);
            auto lo = PGM_SUB_EPS(pos, Epsilon);
            auto hi = PGM_ADD_EPS(pos, Epsilon, n);
            return {pos, lo, hi};
        }

        // Support operator[]
        static inline constexpr bool support_access() {
            return false;
        }

        K operator[](const size_t idx) const {
            throw std::runtime_error("Operator [] is not supported yet");
        }

        /**
         * Returns the number of segments in the last level of the index.
         * @return the number of segments
         */
        size_t segments_count() const {
            return segments.size();
        }

        /**
         * Returns the number of levels of the index.
         * @return the number of levels of the index
         */
        // TODO sum the btree levels or not?
        size_t height() const {
            return 1;
        }

        /**
         * Returns the size of the index in bytes.
         * @return the size of the index in bytes
         */
        size_t size_in_bytes() const {
            return segments.size() * sizeof(SegmentData) + first_keys.size_all_bytes();
        }

        void clear() {
            first_keys.clear();
            segments = std::vector<SegmentData>();
        }
    };
}
