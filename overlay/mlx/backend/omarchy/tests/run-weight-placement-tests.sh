#!/bin/sh
# Compile and run the weight placement tests. Dependency: a C++17 compiler
# (g++ or clang++) and the Vulkan headers. Exits nonzero on the first
# failing expectation.
# no set -e: POSIX sh has no pipefail; set -eu is set and there are no pipelines
set -eu
cd "$(dirname "$0")"
CXX=${CXX:-g++}
OUT=$(mktemp /tmp/omarchy-weight-placement-test.XXXXXX)
trap 'rm -f "$OUT"' EXIT
$CXX -std=c++17 -O0 -Wall -Wextra -I../.. -o "$OUT" weight_placement_test.cpp
"$OUT"
