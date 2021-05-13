#!/bin/bash

pacman --noconfirm -S git
pacman --noconfirm -S mingw-w64-x86_64-fftw
pacman --noconfirm -S mingw-w64-x86_64-openblas
pacman --noconfirm -S mingw-w64-x86_64-libpng

cd /
echo "Downloading POSIX.1b Realtime Extensions library (librt)"
wget --quiet https://repo.msys2.org/msys/x86_64/msys2-runtime-devel-3.2.0-3-x86_64.pkg.tar.zst
echo "Installing `tar -I zstd -xvf msys2-runtime-devel-3.2.0-3-x86_64.pkg.tar.zst usr/lib/librt.a`"
rm msys2-runtime-devel-3.2.0-3-x86_64.pkg.tar.zst

GCC_PATH="/mingw64/bin"
if [ -d "$GCC_PATH" ] && [[ ":$PATH:" != *":$GCC_PATH:"* ]]; then
    echo "export PATH=$GCC_PATH:\$PATH" >> ~/.bashrc
fi
