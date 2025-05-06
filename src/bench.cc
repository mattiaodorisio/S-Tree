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
#include "b_tree_pgm.h"

#define DATA_INT
#ifdef DATA_DOUBLE
typedef double data_t;
#else
typedef uint32_t data_t;
#endif

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
      std::vector<uint64_t> file_dataset = load_data<uint64_t>(data == REAL_WORLD_1 ? "../data/books_200M_uint32" : "../data/wiki_ts_200M_uint64", input_size);
      
      // Copy in v (makes the cast)
      v.resize(file_dataset.size());
      std::copy(file_dataset.begin(), file_dataset.end(), v.begin());
      
      if (v.size() < input_size)
        input_size = v.size();
      
      if (!std::is_sorted(v.begin(), v.end()))
        ips4o::parallel::sort(v.begin(), v.end());

      v.resize(input_size);

      break;
    }
  }

  // For some index the max value is reserved as a sentinel, decrease the last values of the array if equal to max
  for (size_t i = v.size() - 1; i >= 0 && v[i] == std::numeric_limits<data_t>::max(); --i)
    --v[i];

  return v;
}


bool warning = false;

class SearchFixture : public ::benchmark::Fixture {
 public:
  SearchFixture() {
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
    q.resize(QUERY_SIZE);
    std::generate(q.begin(), q.end(), [&]() { return dis(gen); });

    // Drop the cache
    if (warning == false && std::system("sudo sh -c \"sync; echo 1 > /proc/sys/vm/drop_caches\"") != 0) {
      std::cerr << "WARNING: cannot clear the cache before starting the benchmark" << std::endl;
      warning = true;
    }
  }

  void TearDown(::benchmark::State& state) {
    vec = std::vector<data_t>(0);
    q = std::vector<data_t>(0);
  }

  std::vector<data_t> vec;
  std::vector<data_t> q;
};

template <typename Index>
class index_fixture : public SearchFixture {
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

  void TearDown(::benchmark::State& state) {
    SearchFixture::TearDown(state);
    e.clear();
  }

  Index e;
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
  BENCHMARK_REGISTER_F(fixture, bench_name)->ArgsProduct({{0, 1, 2, 3}, {1000, 100'000, 50'000'000, 200'000'000}});

// Macros to use templates within macros, parenthesis are needed
// The template type t must be referred as argument_type<void(idx)>::type within the macro
template<typename T> struct argument_type;
template<typename T, typename U> struct argument_type<T(U)> { typedef U type; };

#define SEARCH_BENCHMARK_T(fixture, bench_name, fun, idx) \
  BENCHMARK_TEMPLATE_DEFINE_F(fixture, bench_name, argument_type<void(idx)>::type)(benchmark::State& st) { \
    size_t i = 0; \
    for (auto _ : st) { \
      auto res = fun(q[i++ % QUERY_SIZE]); \
      benchmark::DoNotOptimize(res); \
      benchmark::ClobberMemory(); \
      __sync_synchronize(); \
    } \
  } \
  BENCHMARK_REGISTER_F(fixture, bench_name)->ArgsProduct({{0, 1, 2, 3}, {1000, 100'000, 50'000'000, 200'000'000}});

#define SUM_BENCHMARK_NT(fixture, bench_name) \
  BENCHMARK_DEFINE_F(fixture, bench_name)(benchmark::State& st) { \
    for (auto _ : st) { \
      data_t sum = 0; \
      for (size_t i = 0; i < vec.size(); ++i) \
        sum += vec[i]; \
      benchmark::DoNotOptimize(sum); \
      benchmark::ClobberMemory(); \
      __sync_synchronize(); \
    } \
  } \
  BENCHMARK_REGISTER_F(fixture, bench_name)->ArgsProduct({{0, 1, 2, 3}, {100'000, 100'000'000}});

#define SUM_BENCHMARK_T(fixture, bench_name, idx) \
  BENCHMARK_TEMPLATE_DEFINE_F(fixture, bench_name, argument_type<void(idx)>::type)(benchmark::State& st) { \
    for (auto _ : st) { \
      data_t sum = 0; \
      for (size_t i = 0; i < e.size(); ++i) \
        sum += e[i]; \
      benchmark::DoNotOptimize(sum); \
      benchmark::ClobberMemory(); \
      __sync_synchronize(); \
    } \
  } \
  BENCHMARK_REGISTER_F(fixture, bench_name)->ArgsProduct({{0, 1, 2, 3}, {100'000, 100'000'000}});

#if false
SUM_BENCHMARK_NT(SearchFixture, sum_vec);
SUM_BENCHMARK_T(index_fixture, sum_ebs, SIMD_Btree::ebs<data_t>);
SUM_BENCHMARK_T(index_fixture, sum_b_tree_256_1, (SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX2, 1>));
#endif

// Resgister the benchmark
SEARCH_BENCHMARK_NT(SearchFixture, std_lower_bound, std::lower_bound);
SEARCH_BENCHMARK_T(index_fixture, ebs, e.lower_bound_idx, SIMD_Btree::ebs<data_t>);

SEARCH_BENCHMARK_T(index_fixture, b_tree_256_1, e.lower_bound_idx, (SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX2, 1>));
SEARCH_BENCHMARK_T(index_fixture, b_tree_256_2, e.lower_bound_idx, (SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX2, 2>));
SEARCH_BENCHMARK_T(index_fixture, b_tree_256_4, e.lower_bound_idx, (SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX2, 4>));
SEARCH_BENCHMARK_T(index_fixture, b_tree_512_1, e.lower_bound_idx, (SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX512, 1>));
SEARCH_BENCHMARK_T(index_fixture, b_tree_512_2, e.lower_bound_idx, (SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX512, 2>));
SEARCH_BENCHMARK_T(index_fixture, b_tree_512_4, e.lower_bound_idx, (SIMD_Btree::btree<data_t, 0, SIMD_Btree::SIMD_ext::AVX512, 4>));

SEARCH_BENCHMARK_T(index_fixture, b_plus_256_1_1, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 1, 1>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_256_1_2, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 1, 2>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_256_1_4, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 1, 4>));

SEARCH_BENCHMARK_T(index_fixture, b_plus_256_2_1, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 2, 1>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_256_2_2, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 2, 2>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_256_2_4, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 2, 4>));

SEARCH_BENCHMARK_T(index_fixture, b_plus_256_4_1, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 4, 1>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_256_4_2, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 4, 2>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_256_4_4, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX2, 4, 4>));

SEARCH_BENCHMARK_T(index_fixture, b_plus_512_1_1, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 1, 1>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_512_1_2, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 1, 2>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_512_1_4, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 1, 4>));

SEARCH_BENCHMARK_T(index_fixture, b_plus_512_2_1, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 2, 1>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_512_2_2, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 2, 2>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_512_2_4, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 2, 4>));

SEARCH_BENCHMARK_T(index_fixture, b_plus_512_4_1, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 4, 1>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_512_4_2, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 4, 2>));
SEARCH_BENCHMARK_T(index_fixture, b_plus_512_4_4, e.lower_bound_idx, (SIMD_Btree::sampled_btree<data_t, SIMD_Btree::SIMD_ext::AVX512, 4, 4>));

SEARCH_BENCHMARK_T(index_fixture, pgm_btree, e.search, (pgm::BTreePGMIndex<data_t>));

// Run the benchmark
BENCHMARK_MAIN();
