#include <iostream>
#include <vector>

#include "b_tree.h"

int main() {

    std::vector<int32_t> data = {-3, 2, 4, 11, 35, 60};
    SIMD_Btree::btree<int32_t> tree;
    tree.build(data.begin(), data.end());
    size_t pos11 = tree.search(11);
    size_t pos12 = tree.search(12);

    std::cout << "Data: -3, 2, 4, 11, 35, 60" << std::endl;
    std::cout << "The lower bound of 11 is at index " << pos11 << ", the lower bound of 12 is at index " << pos12 << std::endl;
    return 0;

}
