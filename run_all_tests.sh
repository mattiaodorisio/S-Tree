#!/bin/bash

# This is a temporary script intended to be run from the build directory

set -e # Always exit on error

make -j tests*
./tests_uint32
./tests_uint64
./tests_double
./tests_int32
./tests_int64
./tests_float
