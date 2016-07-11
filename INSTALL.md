OVERVIEW
=========

BART depends upon the new lapacke.h and cblas.h interfaces for blas and lapack. These
new 'C' interfaces are often not packaged robustly on many different platforms.
Important fixes to lapack distribution occured at lapack-3.6.0 and for finding
lapack libs robustly with the github patch version 7fb63b1cd386b099d7da6eeaafc3e7dce055a7d0.

List of issues when using default lapack from distributions:
 1) Homebrew provides lapacke.h, but not cblas.h
 2) SUSE provides cblas.h from ATLAS, but lapacke.h from lapack distro
 3) RHEL6 does not provide lapacke.h or cblas.h
 4) RHEL7 does not provide lapacke.h or cblas.h

Additionally, several 'Vendor' optimized versions provide enhanced lapack and blas intefaces with various levels of support for lapacke.h and cblas.h.


export BLD_DIR=~/src
export CC=gcc               # or clang
export CXX=g++              # or clang++
export FC=$(which gfortran) # Make sure that C, C++, and Fortran compilers are compatible with each other!


LAPACKE (LAPACK + C interfaces) BUILD HINTS:
============================================
cd ${BLD_DIR}
git clone https://github.com/Reference-LAPACK/lapack.git
cd ${BLD_DIR}/lapack
git checkout 7fb63b1cd386b099d7da6eeaafc3e7dce055a7d0 -b Fixed64BitLibFinding
mkdir -p ${BLD_DIR}/lapack-bld
cd ${BLD_DIR}/lapack-bld
cmake -DCMAKE_Fortran_COMPILER:FILEPATH=${FC} -DCBLAS:BOOL=ON -DLAPACKE:BOOL=ON -DCMAKE_INSTALL_PREFIX:PATH=${BLD_DIR}/lapack-install ../lapack
make -j4
make install


MAC & Linux builds:
======

mkdir -p ${BLD_DIR}/bart-LAPACKE-bld
cd ${BLD_DIR}/bart-LAPACKE-bld
rm -rf *;
CC=clang CXX=clang++ cmake -DCMAKE_Fortran_COMPILER:FILEPATH=${FC} -DLINALG_VENDOR=LAPACKE -DLAPACKE_DIR=${BLD_DIR}/lapack-install ../bart
make -j23

-- OR --
mkdir -p ${BLD_DIR}/bart-OpenBLAS-bld
cd ${BLD_DIR}/bart-OpenBLAS-bld
rm -rf *;
CC=clang CXX=clang++ cmake -DCMAKE_Fortran_COMPILER:FILEPATH=${FC} -DLINALG_VENDOR=OpenBLAS -DOpenBLAS_DIR=${BLD_DIR}/lapack-install ../bart
make -j23


