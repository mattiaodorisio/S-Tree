## SIMD-B-Tree
This is the implementation of a static pointerless and SIMD B-Tree, inspired from:

1. [This post](https://en.algorithmica.org/hpc/data-structures/s-tree/) on [algorithmica.org](https://en.algorithmica.org/)
1. The _CSS-Tree_ presented in the [paper](https://dl.acm.org/doi/10.5555/645925.671362) _Cache conscious indexing for decision-support in main memory_ by Rao, J., & Ross, K. A. (1998), and [its implementation](https://github.com/gvinciguerra/CSS-tree) by [@gvinciguerra](https://github.com/gvinciguerra)

## Usage example

```c++
#include "b_tree.h"

std::vector<int32_t> data = {-3, 2, 4, 11, 35, 60};
SIMD_Btree::btree<int32_t> tree;
tree.build(data.begin(), data.end());
size_t rank = tree.search(11);
```

Look also at `src/example.cc` for a working example.
