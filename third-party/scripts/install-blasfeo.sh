#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"/..

build_type="${1:-Release}"
target="${2:-X64_INTEL_HASWELL}"
version="master"

set -ex
prefix="$PWD/staging/blasfeo-$target"
export CMAKE_PREFIX_PATH="$prefix:$CMAKE_PREFIX_PATH"

mkdir -p src
pushd src

# BLASFEO
if [ ! -d blasfeo ]; then
    git clone --single-branch --depth=1 --branch "$version" \
        https://github.com/giaf/blasfeo.git
    pushd blasfeo; git apply ../../patches/blasfeo-*.patch; popd
fi
pushd blasfeo
CFLAGS="-Wno-error=incompatible-pointer-types" \
cmake -S. -Bbuild-$target \
    -G "Ninja" \
    -D CMAKE_INSTALL_PREFIX="$prefix" \
    -D CMAKE_BUILD_TYPE=$build_type \
    -D TARGET=$target \
    -D BLASFEO_EXAMPLES=Off \
    --toolchain "$HOME/opt/gcc14/x-tools/x86_64-bionic-linux-gnu.toolchain.cmake"
cmake --build build-$target -j
cmake --install build-$target
popd

popd
