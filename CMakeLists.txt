cmake_minimum_required(VERSION 3.13)
project(Search)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -fopt-info-vec-optimized")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}  -O3 -Wall -g -march=native")
set(CMAKE_BUILD_TYPE RELEASE)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

# Dependencies
#add_subdirectory(${PROJECT_SOURCE_DIR}/extern/googletest)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/benchmark)
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/ips4o)

# Include dir
include_directories(${PROJECT_SOURCE_DIR}/include)

# Example
set(EXAMPLE example)
add_executable(${EXAMPLE} src/example.cc)

# Main benchmark
set(MAIN_BENCH bench)
add_executable(${MAIN_BENCH} src/bench.cc)
target_link_libraries(${MAIN_BENCH} PRIVATE benchmark)
target_link_libraries(${MAIN_BENCH} PRIVATE ips4o)

# Tests
set(TESTS_DOUBLE tests_double)
add_executable(${TESTS_DOUBLE} src/tests.cc)
target_link_libraries(${TESTS_DOUBLE} PRIVATE ips4o)
target_compile_definitions(${TESTS_DOUBLE} PRIVATE TEST_DOUBLE)

set(TESTS_FLOAT tests_float)
add_executable(${TESTS_FLOAT} src/tests.cc)
target_link_libraries(${TESTS_FLOAT} PRIVATE ips4o)
target_compile_definitions(${TESTS_FLOAT} PRIVATE TEST_FLOAT)

set(TESTS_UINT32 tests_uint32)
add_executable(${TESTS_UINT32} src/tests.cc)
target_link_libraries(${TESTS_UINT32} PRIVATE ips4o)
target_compile_definitions(${TESTS_UINT32} PRIVATE TEST_UINT32)

set(TESTS_INT32 tests_int32)
add_executable(${TESTS_INT32} src/tests.cc)
target_link_libraries(${TESTS_INT32} PRIVATE ips4o)
target_compile_definitions(${TESTS_INT32} PRIVATE TEST_INT32)

set(TESTS_UINT64 tests_uint64)
add_executable(${TESTS_UINT64} src/tests.cc)
target_link_libraries(${TESTS_UINT64} PRIVATE ips4o)
target_compile_definitions(${TESTS_UINT64} PRIVATE TEST_UINT64)

set(TESTS_INT64 tests_int64)
add_executable(${TESTS_INT64} src/tests.cc)
target_link_libraries(${TESTS_INT64} PRIVATE ips4o)
target_compile_definitions(${TESTS_INT64} PRIVATE TEST_INT64)
