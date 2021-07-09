#!/bin/bash

pacman --sync --noconfirm --needed base-devel mingw-w64-x86_64-toolchain git mingw-w64-x86_64-fftw mingw-w64-x86_64-openblas mingw-w64-x86_64-libpng

echo "Installing /usr/lib/librt.a"
CURRENT_PATH=$(pwd)
cd /
curl https://repo.msys2.org/msys/x86_64/msys2-runtime-devel-3.2.0-3-x86_64.pkg.tar.zst | tar -I zstd -x usr/lib/librt.a
cd $CURRENT_PATH

GCC_PATH="/mingw64/bin"
if [ -d "$GCC_PATH" ] && [[ ":$PATH:" != *":$GCC_PATH:"* ]]; then
    echo "export PATH=$GCC_PATH:\$PATH" >> ~/.bashrc
fi
