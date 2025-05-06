# S-Tree
S-Tree is a Static pointer-free and SIMD-optimized B-Tree, inspired by [this post](https://en.algorithmica.org/hpc/data-structures/s-tree/) on [algorithmica.org](https://en.algorithmica.org/)

## Overview

The S-Tree library offers efficient data structures optimized for modern hardware. It uses cache-sensitive layouts and SIMD intrinsics to enhance the search performance, particularly for computing the rank (i.e. lower_bound) of elements, making it ideal for read-heavy workloads.

Other useful references and similar projects:

1. The Static Search Tree ([post](https://curiouscoding.nl/posts/static-search-tree/) and [repo](https://github.com/RagnarGrootKoerkamp/suffix-array-searching)) written in Rust by [@RagnarGrootKoerkamp](https://github.com/RagnarGrootKoerkamp), with common ideas plus more complex layouts and batched queries.
1. FAST: fast architecture sensitive tree, presented in this [paper](https://dl.acm.org/doi/10.1145/1807167.1807206)
1. The _CSS-Tree_ presented in the [paper](https://dl.acm.org/doi/10.5555/645925.671362) _Cache conscious indexing for decision-support in main memory_ by Rao, J., & Ross, K. A. (1998), and its [implementation](https://github.com/gvinciguerra/CSS-tree) by [@gvinciguerra](https://github.com/gvinciguerra)

## Data Structures

### BTree
The `BTree` is a pointer-free B-Tree (not a B+Tree) that organizes elements in a cache-sensitive layout similar to the CSS Tree. It generalizes the [Eytzinger layout](https://onlinelibrary.wiley.com/doi/abs/10.1002/spe.3150) (commonly used for binary trees) to support a configurable fanout `B` (node size). Key features include:

- **Efficient Traversal**: Uses SIMD intrinsics for fast operations.
- **Space Efficiency**: The input data is permuted into the BTree layout, allowing the original sorted input to be discarded.
- **Performance Trade-Off**: Accessing elements incurs a small overhead of $\Theta(\log_B n)$ due to the layout transformation.

Additional optimizations inspired by the post include cache-line alignment and the use of [huge pages](https://wiki.debian.org/Hugepages) to minimize [TLB](https://it.wikipedia.org/wiki/Translation_Lookaside_Buffer) misses.

### Sampled-BTree
The `Sampled-BTree` is a variant of the `BTree` that operates on a subset of the input data, extracted via regular sampling with a fixed step. This structure proved to be more efficient on some input.

## Usage Example

```c++
#include "b_tree.h"

std::vector<int32_t> data = {-3, 2, 4, 11, 35, 60};
SIMD_Btree::btree<int32_t> tree;
tree.build(data.begin(), data.end());
std::cout << tree.lower_bound_idx(11) << std::endl;  // 3
std::cout << tree.lower_bound_idx(12) << std::endl;  // 4
std::cout << tree[3] << std::endl;                   // 11
```

For a complete working example, refer to `src/example.cc`.

## Additional Details

When declaring a `btree` or `sampled_btree`, several template parameters can be specified (all have default values):

```
template <typename key_type,
          size_t known_log_tree_size = 0,
          SIMD_ext ext = AVX512,
          size_t vectors_per_block = 1>
class btree;

template <typename key_type,
          SIMD_ext ext = AVX512,
          size_t vectors_per_block = 1,
          size_t leaves_vectors_per_block = vectors_per_block>
class sampled_btree;
```

### Template Parameters

- **`vectors_per_block`**: Specifies the number of vectors per tree node.
- **`ext`**: Defines the SIMD extension to use (`AVX2` or `AVX512`).
- **`leaves_vectors_per_block`**: Sets the sampling offset for the `sampled_btree`.
- **`known_log_tree_size`**: If the logarithm (base `B`) of the data size is known at compile time (consider using `btree::log_size(size_t n)`) this allows additional optimizations. The default value is `0` (unknown).

### Benchmarks
This library was developed as part of the experimental setup for a research article that compares various indexing structures. For a detailed experimental evaluation and performance comparison have a look at the [repo](https://github.com/LorenzoBellomo/SortedStaticIndexBenchmark) and the [paper]().
