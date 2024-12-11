#include <benchmark/benchmark.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdlib>
#include "utils.h"
#include "ips4o.hpp"
#include "ebs.h"
#include "b_tree.h"
#include "sampled_b_tree.h"

#define DATA_INT
#ifdef DATA_DOUBLE
typedef double data_t;
#else
typedef uint32_t data_t;
#endif

//constexpr size_t INPUT_SIZE = std::pow(8, 9) - 1; // 2^27=8^9 -> ok for both ebs and ebs_line
constexpr size_t INPUT_SIZE = 100'000'000; // 2^27=8^9 -> ok for both ebs and ebs_line
constexpr size_t QUERY_SIZE = 1000000;

enum dataset {
  UNIFORM,
  EXPONENTIAL,
  REAL_WORLD_1, // BOOKS
  REAL_WORLD_2  // WIKI
};

template<typename Key>
std::vector<Key> generate_vec(const dataset data, size_t input_size) {
  std::vector<Key> v;
  switch (data) {
    case UNIFORM: {
      v.resize(input_size);
      constexpr uint_fast32_t seed = 42;
      std::mt19937 gen(seed);
#ifdef DATA_DOUBLE
      std::uniform_real_distribution<data_t> dis(0.0, 1.0);
#elif defined DATA_INT
      std::uniform_int_distribution<data_t> dis;
#else
#error Unspecified data type
#endif
      std::generate(v.begin(), v.end(), [&]() { return dis(gen); });
      ips4o::parallel::sort(v.begin(), v.end());
      break;
    }

    case EXPONENTIAL: {
      v.resize(input_size);
      constexpr uint_fast32_t seed = 42;
      std::mt19937 gen(seed);
      std::exponential_distribution<double> dis;
      std::generate(v.begin(), v.end(), [&]() { return static_cast<data_t>(std::round(dis(gen) * (std::numeric_limits<data_t>::max() / 2))) ; });
      ips4o::parallel::sort(v.begin(), v.end());
      break;
    }

    case REAL_WORLD_1:
    case REAL_WORLD_2:
    {
      v = load_data<uint32_t>(data == REAL_WORLD_1 ? "../../SOSD/data/books_100M_uint32" : "../../SOSD/data/wiki_ts_100M_uint32");
      if (v.size() < input_size)
        input_size = v.size();
      
      if (!std::is_sorted(v.begin(), v.end()))
        ips4o::parallel::sort(v.begin(), v.end());

      v.resize(input_size);

      break;
    }
  }

  return v;
}


bool warning = false;

class SearchFixture : public ::benchmark::Fixture {
 public:
  SearchFixture() {
    q.resize(QUERY_SIZE);
  }

  ~SearchFixture() {
  }

  void SetUp(const ::benchmark::State& state) {
    vec = generate_vec<data_t>(static_cast<const dataset>(state.range(0)), state.range(1));

    // Generate the query vector (always uniform distr)
#ifdef DATA_DOUBLE
    std::uniform_real_distribution<data_t> dis(vec.front(), vec.back());
#elif defined DATA_INT
    std::uniform_int_distribution<data_t> dis(vec.front(), vec.back());
#else
#error Unspecified data type
#endif
    constexpr uint_fast32_t seed = 42;
    std::mt19937 gen(seed);
    std::generate(q.begin(), q.end(), [&]() { return dis(gen); });

    // Drop the cache
    if (warning == false && std::system("sudo sh -c \"sync; echo 1 > /proc/sys/vm/drop_caches\"") != 0) {
      std::cerr << "WARNING: cannot clear the cache before starting the benchmark" << std::endl;
      warning = true;
    }
  }

  void TearDown(::benchmark::State& state) {
    // Nothing to do
  }

  std::vector<data_t> vec;
  std::vector<data_t> q;
};

template <SIMD_Btree::SIMD_ext simd_extension, size_t n_vectors>
class b_tree_fixture : public SearchFixture {
public:
  void SetUp(const ::benchmark::State& state) {
    SearchFixture::SetUp(state);
    e.build(vec.data(), vec.data() + vec.size());
    
    // Drop the cache (TODO - this is already done, inefficient)
    if (warning == false && std::system("sudo sh -c \"sync; echo 1 > /proc/sys/vm/drop_caches\"") != 0) {
      std::cerr << "WARNING: cannot clear the cache before starting the benchmark" << std::endl;
      warning = true;
    }
  }

  SIMD_Btree::btree<data_t, simd_extension, n_vectors> e;
};

template <SIMD_Btree::SIMD_ext simd_extension, size_t n_vectors, size_t n_sampled_vector = n_vectors>
class sampled_b_tree_fixture : public SearchFixture {
public:
  void SetUp(const ::benchmark::State& state) {
    SearchFixture::SetUp(state);
    e.build(vec.data(), vec.data() + vec.size());
    
    // Drop the cache (TODO - this is already done, inefficient)
    if (warning == false && std::system("sudo sh -c \"sync; echo 1 > /proc/sys/vm/drop_caches\"") != 0) {
      std::cerr << "WARNING: cannot clear the cache before starting the benchmark" << std::endl;
      warning = true;
    }
  }

  SIMD_Btree::b_plus_tree<data_t, simd_extension, n_vectors, n_sampled_vector> e;
};

#define SEARCH_BENCHMARK_NT(fixture, bench_name, fun) \
  BENCHMARK_DEFINE_F(fixture, bench_name)(benchmark::State& st) { \
    size_t idx = 0; \
    for (auto _ : st) { \
      auto res = fun(vec.data(), vec.data() + vec.size(), q[idx++ % QUERY_SIZE]); \
      benchmark::DoNotOptimize(res); \
      benchmark::ClobberMemory(); \
      __sync_synchronize(); \
    } \
  } \
  BENCHMARK_REGISTER_F(fixture, bench_name)->ArgsProduct({{0, 1, 2, 3}, {1'000'000, 10'000'000, 100'000'000}});

#define SEARCH_BENCHMARK(fixture, bench_name, fun, simd_extension, n_vectors, n_sampled_vector) \
  BENCHMARK_TEMPLATE_DEFINE_F(fixture, bench_name, simd_extension, n_vectors, n_sampled_vector)(benchmark::State& st) { \
    size_t idx = 0; \
    for (auto _ : st) { \
      auto res = fun(q[idx++ % QUERY_SIZE]); \
      benchmark::DoNotOptimize(res); \
      benchmark::ClobberMemory(); \
      __sync_synchronize(); \
    } \
  } \
  BENCHMARK_REGISTER_F(fixture, bench_name)->ArgsProduct({{0, 1, 2, 3}, {1'000'000, 10'000'000, 100'000'000}});

// Resgister the benchmark
SEARCH_BENCHMARK_NT(SearchFixture, std_lower_bound, std::lower_bound);

//SEARCH_BENCHMARK(b_tree_fixture, ebs_line_256_1, e.search, SIMD_Btree::SIMD_ext::AVX2, 1);
//SEARCH_BENCHMARK(b_tree_fixture, ebs_line_256_2, e.search, SIMD_Btree::SIMD_ext::AVX2, 2);
//SEARCH_BENCHMARK(b_tree_fixture, ebs_line_256_4, e.search, SIMD_Btree::SIMD_ext::AVX2, 4);
//SEARCH_BENCHMARK(b_tree_fixture, ebs_line_512_1, e.search, SIMD_Btree::SIMD_ext::AVX512, 1);
//SEARCH_BENCHMARK(b_tree_fixture, ebs_line_512_2, e.search, SIMD_Btree::SIMD_ext::AVX512, 2);
//SEARCH_BENCHMARK(b_tree_fixture, ebs_line_512_4, e.search, SIMD_Btree::SIMD_ext::AVX512, 4);

SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_1_1, e.search, SIMD_Btree::SIMD_ext::AVX2, 1, 1);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_1_2, e.search, SIMD_Btree::SIMD_ext::AVX2, 1, 2);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_1_4, e.search, SIMD_Btree::SIMD_ext::AVX2, 1, 4);

SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_2_1, e.search, SIMD_Btree::SIMD_ext::AVX2, 2, 1);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_2_2, e.search, SIMD_Btree::SIMD_ext::AVX2, 2, 2);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_2_4, e.search, SIMD_Btree::SIMD_ext::AVX2, 2, 4);

SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_4_1, e.search, SIMD_Btree::SIMD_ext::AVX2, 4, 1);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_4_2, e.search, SIMD_Btree::SIMD_ext::AVX2, 4, 2);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_256_4_4, e.search, SIMD_Btree::SIMD_ext::AVX2, 4, 4);

SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_1_1, e.search, SIMD_Btree::SIMD_ext::AVX512, 1, 1);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_1_2, e.search, SIMD_Btree::SIMD_ext::AVX512, 1, 2);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_1_4, e.search, SIMD_Btree::SIMD_ext::AVX512, 1, 4);

SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_2_1, e.search, SIMD_Btree::SIMD_ext::AVX512, 2, 1);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_2_2, e.search, SIMD_Btree::SIMD_ext::AVX512, 2, 2);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_2_4, e.search, SIMD_Btree::SIMD_ext::AVX512, 2, 4);

SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_4_1, e.search, SIMD_Btree::SIMD_ext::AVX512, 4, 1);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_4_2, e.search, SIMD_Btree::SIMD_ext::AVX512, 4, 2);
SEARCH_BENCHMARK(sampled_b_tree_fixture, b_plus_512_4_4, e.search, SIMD_Btree::SIMD_ext::AVX512, 4, 4);

// Run the benchmark
BENCHMARK_MAIN();
