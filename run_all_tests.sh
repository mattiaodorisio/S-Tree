#!/bin/bash

# This script is intended to be launched from the build directory

set -e # Always exit on error

make -j tests_uint32 tests_uint64 tests_double tests_int32 tests_int64 tests_float

./tests_uint32
./tests_uint64
./tests_double
./tests_int32
./tests_int64
./tests_float
