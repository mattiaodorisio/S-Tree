#pragma once

#include <vector>
#include <fstream>

/**
 * Reads a dataset file @p filename in binary format and writes keys to vector.
 * @tparam Key the type of the key
 * @param filename name of the dataset file
 * @return vector of keys
 */
template<typename Key>
std::vector<Key> load_data(const std::string &filename) {
    using key_type = Key;

    // Open file.
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Could not load " << filename << '.' << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read number of keys.
    uint64_t n_keys;
    in.read(reinterpret_cast<char*>(&n_keys), sizeof(uint64_t));

    // Initialize vector.
    std::vector<key_type> data;
    data.resize(n_keys);

    // Read keys.
    in.read(reinterpret_cast<char*>(data.data()), n_keys * sizeof(key_type));
    in.close();

    return data;
}

template <typename ForwardIt>
size_t count_distinct_in_sorted(ForwardIt begin, const ForwardIt end) {
    size_t cnt = 0;
    for (++begin; begin != end; ++begin)
        cnt += (begin[0] != begin[-1]) ? 1 : 0;
    return cnt;
}
