#!/bin/bash


export ROOT_DIR=$PWD

# Define the versions we want (where we expect things to work)
fftw3_version="3.3.6-pl2"
openblas_version="0.2.19"

fftw3_name="fftw-${fftw3_version}"
openblas_name="v${openblas_version}"

# Build OpenBLAS
if [ -f OpenBLAS-${openblas_version}/done ]
then
	echo "skipping OpenBLAS-${openblas_version}"
else
	[ -f openblas_${openblas_name}.tar.gz] || wget -O openblas_${openblas_name}.tar.gz https://github.com/xianyi/OpenBLAS/archive/${openblas_name}.tar.gz
	tar xvzf openblas_${openblas_name}.tar.gz && cd OpenBLAS-${openblas_version}
	CFLAGS="-DUSE_CUDA=1"
 	make -j 12 all 
	make -j 12 netlib
	make PREFIX=${ROOT_DIR}/deps/openblas install
	touch done
	cd -
	sleep 5
fi

# Build FFTW3
if [ -f ${fftw3_name}/done ]
then
	echo "skipping ${fftw3_name}"
else
	[ -f fftw_${fftw3_name}.tar.gz ] || wget -O fftw_${fftw3_name}.tar.gz http://fftw.org/${fftw3_name}.tar.gz
	tar xvzf fftw_${fftw3_name}.tar.gz && cd ${fftw3_name}
	CFLAGS="-fPIC" CPPFLAGS="-fPIC" ./configure --prefix=$ROOT_DIR/deps/fftw --enable-float --enable-threads # Note the tricky flag: --enable-float
	make -j 12  && make install
	touch done
	cd -
	sleep 5
fi


if [ -f ${ROOT_DIR}/lapack-install ]
then
 	echo "lapack installed already." 
else
	git clone https://github.com/Reference-LAPACK/lapack.git ${ROOT_DIR}/src/lapack
	cd ${ROOT_DIR}/src/lapack
	mkdir -p build
	cd build
	CC=gcc CXX=g++ FC=gfortran cmake -DCBLAS=ON -DLAPACKE=ON -DCMAKE_INSTALL_PREFIX=${ROOT_DIR}/lapack-install ..
	make -j4
	make install
fi


export LINALG_VENDOR=OPENBLAS
export LAPACKE_ROOT=${ROOT_DIR}/lapack-install
export LAPACKE_DIR=${ROOT_DIR}/lapack-install
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0
export LAPACKE_CBLAS_LIB=${ROOT_DIR}/lapack-install
export OPENBLAS_ROOT=${ROOT_DIR}/lapack-install
export OPENBLAS_DIR=${ROOT_DIR}/lapack-install

mkdir -p ${ROOT_DIR}/bart-LAPACKE-bld
cd ${ROOT_DIR}/bart-LAPACKE-bld
rm -rf *;
CC=gcc CXX=g++ cmake -DOpenBLAS_DIR=${ROOT_DIR}/deps/openblas -NOLAPACK=ON -DBART_NO_LAPACKE=ON -#BART_FFTWTHREADS=ON -BART_GENERATE_DOC=OFF -DUSE_CUDA=OFF -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0  ..
#CC=gcc CXX=g++ cmake -NOLAPACK=OFF -BART_FFTWTHREADS=ON -BART_GENERATE_DOC=ON -DUSE_CUDA=OFF DCMAKE_Fortran_COMPILER=/usr/bin/gfortran -DOpenBLAS_DIR=${ROOT_DIR}/deps/openblas -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0  ..
make -j8


#mkdir -p ${ROOT_DIR}/bart-LAPACKE-bld
#cd ${ROOT_DIR}/bart-LAPACKE-bld
#CC=gcc CXX=g++ cmake -NOLAPACK=ON -BART_FFTWTHREADS=ON -BART_GENERATE_DOC=ON -DUSE_CUDA=ON DCMAKE_Fortran_COMPILER=/usr/bin/gfortran -DOpenBLAS_DIR=${ROOT_DIR}/deps/openblas -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.0  ..
#make -j8 -d

