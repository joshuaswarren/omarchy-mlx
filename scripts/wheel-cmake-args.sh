#!/usr/bin/env bash
# no set -e: sourced by build-wheel.sh, which sets strict mode
# Base CMake args for the mlx-omarchy release wheel build. Sourced by
# scripts/build-wheel.sh, which appends host-specific additions to
# $CMAKE_ARGS afterwards.
#
# -DCMAKE_INSTALL_LIBDIR=lib pins the library destination to lib/ on
# every host: without it, a build on a libdir-lib64 host (Fedora
# convention) installs libmlx.so into mlx/lib64/ while the extension
# module carries RUNPATH $ORIGIN/lib, and import mlx.core fails on the
# installed wheel (issue 537). scripts/openblas_isolation.py
# check-layout refuses such a wheel at the build.
CMAKE_ARGS="-DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DCMAKE_INSTALL_LIBDIR=lib"
