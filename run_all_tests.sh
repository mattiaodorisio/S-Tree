#!/bin/bash
set -e # Always exit on error

mkdir -p build
cd build
cmake ..
make -j tests_uint32 tests_uint64 tests_double tests_int32 tests_int64 tests_float

./tests_uint32
./tests_uint64
./tests_double
./tests_int32
./tests_int64
./tests_float

cd ..
