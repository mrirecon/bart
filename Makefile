# Copyright 2013-2015. The Regents of the University of California.
# Copyright 2015-2022. Martin Uecker
# Copyright 2022-2024. Institute of Biomedical Imaging, TU Graz.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.

# silent make
#MAKEFLAGS += --silent

# auto clean on makefile updates
AUTOCLEAN?=1

# How it works: older versions of Make will be sorted before 4.0, making the comparison fail
ifneq (4.0,$(firstword $(sort $(MAKE_VERSION) 4.0)))
$(error bart requires version 4.0 of GNU make or newer!)
endif


# clear out all implicit rules
MAKEFLAGS += --no-builtin-rules
# clear out some variables by hand, as we cannot use -R, --no-builtin-variables without recursive make
# but only undefine them if they come from their default values
define undef_builtin
ifeq ($(origin $(1)),default)
undefine $(1)
endif
endef

$(eval $(foreach VAR,CC CXX CPP LD ARFLAGS ,$(eval $(call undef_builtin,$(VAR)))))

# Paths

here  = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
root := $(here)

srcdir = $(root)/src
libdir = $(root)/lib
bindir = $(root)/bin

export LOCKDIR?=${libdir}
export BART_TOOLBOX_PATH=$(root)

MAKEFILES = $(wildcard $(root)/Makefiles/Makefile.*)
ALLMAKEFILES = $(root)/Makefile $(wildcard $(root)/Makefile.* $(root)/*.mk $(root)/rules/*.mk $(root)/Makefiles/Makefile.*)

-include Makefile.local
-include Makefile.$(NNAME)
-include $(MAKEFILES)

# some operations might still be non deterministic
NON_DETERMINISTIC?=0

# allow blas calls within omp regions (fails on Debian 9, openblas)
BLAS_THREADSAFE?=0

# use for ppc64le HPC
MPI?=0
OPENBLAS?=0
MKL?=0
CUDA?=0
CUDNN?=0
ACML?=0
OMP?=1
SLINK?=0
DEBUG?=0
UBSAN?=0
ASAN?=0
FFTWTHREADS?=1
SCALAPACK?=0
ISMRMRD?=0
TENSORFLOW?=0
PYTORCH?=0
NOEXEC_STACK?=0
PARALLEL?=0
PARALLEL_NJOBS?=
FORTRAN?=1
PNG?=1
DEBUG_DWARF?=0
WERROR?=0

LOG_BACKEND?=0
LOG_SIEMENS_BACKEND?=0
LOG_ORCHESTRA_BACKEND?=0
LOG_GADGETRON_BACKEND?=0


# The fix that makes AR_LOCK unnecessary is in GNU Make version 4.4.1
# Since MAKE_VERSION only shows the first two version numbers, we need this test
# to be for 4.5 instead of 4.4.
# But we will just comment it out for now, since GNU Make 4.5 does not exist yet,
# and we do not want any surprises if this happens to break anything
#ifeq (4.5,$(firstword $(sort $(MAKE_VERSION) 4.5)))
#AR_LOCK_NEEDED?=0
#else
AR_LOCK_NEEDED?=1
#endif




DESTDIR ?= /
PREFIX ?= usr/local/

BUILDTYPE = Linux
UNAME = $(shell uname -s)
MNAME = $(shell uname -m)
NNAME = $(shell uname -n)

MYLINK=ln



ifeq ($(UNAME),Darwin)
	BUILDTYPE = MacOSX
	MYLINK = ln -s
endif

ifeq ($(BUILDTYPE), MacOSX)
	MACPORTS ?= 1
else
	MACPORTS ?= 0
endif


ifeq ($(BUILDTYPE), Linux)
	# as the defaults changed on most Linux distributions
	# explicitly specify non-deterministic archives to not break make
	ARFLAGS ?= rsU
else
	ARFLAGS ?= rs
endif


ifeq ($(UNAME),Cygwin)
	BUILDTYPE = Cygwin
	NOLAPACKE ?= 1
endif

ifeq ($(UNAME),CYGWIN_NT-10.0)
	BUILDTYPE = Cygwin
	NOLAPACKE ?= 1
endif


ifneq (,$(findstring MSYS,$(UNAME)))
	BUILDTYPE = MSYS
	#LDFLAGS += -lucrtbase # support for %F, %T formatting codes in strftime()
	#LDFLAGS += -static-libgomp
	NOLAPACKE ?= 1
	SLINK = 1
endif



ifeq ($(CC),emcc)
	BUILDTYPE = WASM
endif


HAVE_NOEXECWARN := $(shell ld --help 2>&1 | grep -c "\-no\-warn\-execstack")
ifneq ($(BUILDTYPE), WASM)
ifneq ($(HAVE_NOEXECWARN), 0)
	LDFLAGS += -Wl,-no-warn-execstack
endif
endif

# Automatic dependency generation

DEPFILE = $(*D)/.$(*F).d
DEPFLAG = -MMD -MF $(DEPFILE)
ALLDEPS = $(shell find $(srcdir) utests -name ".*.d")


# Compilation flags
ifeq ($(DEBUG_DWARF),1)
DEBUG=1
endif

ifneq ($(DEBUG),1)
	OPT = -O2
else
	OPT = -Og
endif
#OPT += -ffp-contract=off
CPPFLAGS ?= -Wall -Wextra
CFLAGS ?= $(OPT) -Wmissing-prototypes -Wincompatible-pointer-types -Wsign-conversion -Wwrite-strings
CXXFLAGS ?= $(OPT)

ifeq ($(BUILDTYPE), MacOSX)
	CC ?= gcc-mp-12
else
	CC ?= gcc
ifneq ($(BUILDTYPE), MSYS)
# for symbols in backtraces
	LDFLAGS += -rdynamic
endif
endif


# for debug backtraces
ifeq ($(DEBUG_DWARF),1)
LIBS += -ldw -lunwind
CPPFLAGS += -DUSE_DWARF
endif

ifeq ($(WERROR),1)
CFLAGS += -Werror
endif


ifeq ($(MNAME),riscv64)
	CFLAGS+=-ffp-contract=off
endif


# openblas

ifeq ($(BUILDTYPE), MSYS)
	BLAS_BASE ?= /mingw64/include/OpenBLAS/
else
ifneq ($(BUILDTYPE), MacOSX)
	BLAS_BASE ?= /usr/
else
ifeq ($(MACPORTS),1)
	BLAS_BASE ?= /opt/local/
	CPPFLAGS += -DUSE_MACPORTS
endif
	BLAS_BASE ?= /usr/local/opt/openblas/
endif
endif

ifeq ($(BUILDTYPE), Linux)
ifneq ($(OPENBLAS), 1)
ifneq (,$(findstring Red Hat,$(shell gcc --version)))
	CPPFLAGS+=-I/usr/include/lapacke/
	LDFLAGS+=-L/usr/lib64/atlas -ltatlas
endif
endif
endif

# cuda

CUDA_BASE ?= /usr/
CUDA_LIB ?= lib
CUDNN_BASE ?= $(CUDA_BASE)
CUDNN_LIB ?= lib64

# tensorflow
TENSORFLOW_BASE ?= /usr/local/

# pytorch
PYTORCH_BASE ?= /usr/

# acml

ACML_BASE ?= /usr/local/acml/acml4.4.0/gfortran64_mp/

# mkl
MKL_BASE ?= /opt/intel/mkl/lib/intel64/


# fftw
ifneq ($(BUILDTYPE), MacOSX)
FFTW_BASE ?= /usr/
else
FFTW_BASE ?= /opt/local/
endif



# ISMRM

ISMRM_BASE ?= /usr/local/ismrmrd/



# Main build targets
#
TBASE+=show slice crop resize join transpose squeeze flatten zeros ones flip circshift extract repmat bitmask reshape version delta copy casorati vec poly index multicfl tee compress
TFLP+=scale invert conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg cabs zexp calc unwrap
TNUM+=fft fftmod fftshift noise bench threshold conv rss filter nlmeans mandelbrot wavelet window var std fftrot roistat pol2mask conway morphop hist gmm
TRECO+=pics pocsense sqpics itsense nlinv moba nufft nufftbase rof tgv ictv sake wave lrmatrix estdims estshift estdelay wavepsf wshfl rtnlinv mobafit mobasig grog denoise estscaling
TCALIB+=ecalib ecaltwo caldir walsh cc ccapply rovir calmat svd estvar whiten rmfreq ssa bin psf ncalib phasepole
TMRI+=homodyne poisson twixread fakeksp looklocker upat fovshift seq
TSIM+=phantom traj signal epg sim pulse raga stl bloch grid trajcor coils
TIO+=tee toimg toraw multicfl
TNN+=reconet nnet onehotenc measure mnist tensorflow nlinvnet sample cunet
TMOTION+=affinereg interpolate estmotion

TBASE:=$(sort $(TBASE))
TFLP:=$(sort $(TFLP))
TNUM:=$(sort $(TNUM))
TRECO:=$(sort $(TRECO))
TCALIB:=$(sort $(TCALIB))
TMRI:=$(sort $(TMRI))
TSIM:=$(sort $(TSIM))
TIO:=$(sort $(TIO))
TNN:=$(sort $(TNN))
TMOTION:=$(sort $(TMOTION))



MODULES = -lnum -lmisc -lnum -lmisc
ifeq ($(BUILDTYPE), MSYS)
MODULES += -lwin
endif

MODULES_pics = -lgrecon -lsense -lmotion -liter -llinops -lwavelet -llowrank -lnoncart -lnn -lnlops
MODULES_sqpics = -lsense -liter -llinops -lwavelet -llowrank -lnoncart -llinops
MODULES_pocsense = -lsense -liter -llinops -lwavelet
MODULES_nlinv = -lnoir -lgrecon -lwavelet -llowrank -lnn -liter -lnlops -llinops -lnoncart
MODULES_ncalib = -lnoir -lgrecon -lwavelet -llowrank -lnn -liter -lnlops -llinops -lnoncart
MODULES_phasepole = -lnoir -lgrecon -lwavelet -llowrank -lnn -liter -lnlops -llinops -lnoncart
MODULES_rtnlinv = -lnoir -liter -lnlops -llinops -lnoncart
MODULES_moba = -lmoba -lnoir -lnn -lnlops -llinops -lwavelet -lnoncart -lseq -lstl -lsimu -lgrecon -llowrank -llinops -liter -lnn
MODULES_mobafit = -lmoba -lnlops -llinops -lseq -lstl -lsimu -liter -lnoir
MODULES_bpsense = -lsense -lnoncart -liter -llinops -lwavelet
MODULES_itsense = -liter -llinops
MODULES_ecalib = -lcalib -llinops
MODULES_ecaltwo = -lcalib -llinops
MODULES_estdelay = -lcalib -lnoncart
MODULES_caldir = -lcalib -llinops
MODULES_walsh = -lcalib -llinops
MODULES_calmat = -lcalib -llinops
MODULES_cc = -lcalib -llinops
MODULES_ccapply = -lcalib -llinops
MODULES_estvar = -lcalib -llinops
MODULES_nufft = -lnoncart -liter -llinops
MODULES_rof = -liter -llinops
MODULES_tgv = -liter -llinops
MODULES_ictv = -liter -llinops
MODULES_denoise = -lgrecon -liter -llinops -lwavelet -llowrank -lnoncart -lnn -lnlops
MODULES_bench = -lwavelet -llinops
MODULES_phantom = -lstl -lgeom -lsimu -lgeom
MODULES_bart = -lbox -lgrecon -lsense -lnoir -liter -llinops -lwavelet -llowrank -lnoncart -lcalib -llinops -lseq -lstl -lsimu -lsake -lnlops -lnetworks -lnoir -lnn -liter -lmoba -lgeom -lnn  -lmotion -lnlops
MODULES_sake = -lsake
MODULES_traj = -lnoncart
MODULES_grid = -lstl -lsimu
MODULES_coils = -lstl -lsimu -lgeom
MODULES_raga = -lnoncart -lseq
MODULES_wave = -liter -lwavelet -llinops -llowrank
MODULES_threshold = -llowrank -liter -llinops -lwavelet
MODULES_fakeksp = -lsense -llinops
MODULES_lrmatrix = -llowrank -liter -llinops -lnlops
MODULES_estdims =
MODULES_ismrmrd = -lismrm
MODULES_wavelet = -llinops -lwavelet
MODULES_wshfl = -lgrecon -lsense -liter -llinops -lwavelet -llowrank -lnoncart -lnlops -lnn -lnlops
MODULES_ssa = -lcalib -llinops
MODULES_bin = -lcalib -llinops
MODULES_signal = -lstl -lsimu
MODULES_pol2mask = -lgeom
MODULES_epg = -lstl -lsimu
MODULES_reconet = -lgrecon -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_mnist = -lnetworks -lnn -lnlops -llinops -liter
MODULES_nnet = -lgrecon -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_tensorflow = -lnn -lnlops -llinops -liter
MODULES_sample = -lgrecon -lsense -lmotion -lnetworks -lnn -lnlops -lnoncart -llinops -liter
MODULES_cunet = -lnetworks -lgrecon -lnn -lnlops -llinops -liter
MODULES_measure = -lgrecon -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_onehotenc = -lnn
MODULES_sim = -lseq -lstl -lsimu
MODULES_morphop = -lnlops -llinops -lgeom
MODULES_psf = -lnoncart -llinops
MODULES_nlinvnet = -lnetworks -lnoir -liter -lnn -lnlops -llinops -lnoncart -lgrecon -lnetworks -lsense -liter -llinops -lwavelet -llowrank -lnoncart -lnlops -lnn
MODULES_grog = -lcalib -llinops
MODULES_affinereg = -lmotion -liter -lnlops -llinops
MODULES_estmotion = -lmotion -lnn -liter -lnlops -llinops
MODULES_interpolate = -lmotion -liter -lnlops -llinops
MODULES_stl = -lstl
MODULES_estscaling = -lsense -llinops
MODULES_pulse = -lseq
MODULES_bloch = -lseq -lstl -lsimu
MODULES_trajcor = -lcalib -lnoncart -llinops
MODULES_seq = -lseq -lnoncart -lstl -lsimu
MODULES_mobasig = -lmoba -lnlops -llinops -lnoir -lsimu -lseq

GCCVERSION12 := $(shell expr `$(CC) -dumpversion | cut -f1 -d.` \>= 12)
GCCVERSION14 := $(shell expr `$(CC) -dumpversion | cut -f1 -d.` \>= 14)

# clang

ifeq ($(findstring clang,$(CC)),clang)
CFLAGS += -fblocks
LDFLAGS += -lBlocksRuntime
ifeq ($(DEBUG_DWARF),1)
CFLAGS += -gdwarf -gdwarf-aranges
endif
# Make complains if $(error ...) is indented by tab:
ifeq ($(MPI),1)
$(error ERROR MPI is not support with clang, please compile with gcc)
endif
else ifeq ($(findstring emcc,$(CC)),emcc)
CFLAGS += -fblocks
else
# only add if not clang, as it doesn't understand this:
ifeq ($(GCCVERSION14), 1)
CFLAGS += -Wuseless-cast
else
ifeq ($(GCCVERSION12), 1)
CFLAGS += -Wno-vla-parameter -Wno-nonnull -Wno-maybe-uninitialized
else
$(error ERROR: GCC version 12 or newer is required)
endif
endif
endif

CXX ?= g++
LINKER ?= $(CC)



ifeq ($(ISMRMRD),1)
TMRI += ismrmrd
MODULES_bart += -lismrm
endif

ifeq ($(NOLAPACKE),1)
CPPFLAGS += -DNOLAPACKE
MODULES += -llapacke
endif

ifeq ($(TENSORFLOW),1)
CPPFLAGS += -DTENSORFLOW -I$(TENSORFLOW_BASE)/include
LIBS += -L$(TENSORFLOW_BASE)/lib -Wl,-rpath $(TENSORFLOW_BASE)/lib -ltensorflow_framework -ltensorflow
endif

ifeq ($(PYTORCH),1)
CPPFLAGS += -DPYTORCH -I$(PYTORCH_BASE)/include/torch/csrc/api/include/ -I$(PYTORCH_BASE)/include/
LIBS += -L$(PYTORCH_BASE)/lib -Wl,--no-as-needed,-rpath $(PYTORCH_BASE)/lib -ltorch -ltorch_cpu -lc10
ifeq ($(CUDA),1)
LIBS += -ltorch_cuda -lc10_cuda
endif
endif



XTARGETS += $(TBASE) $(TFLP) $(TNUM) $(TIO) $(TRECO) $(TCALIB) $(TMRI) $(TSIM) $(TNN) $(TMOTION)
XTARGETS:=$(sort $(XTARGETS))

# CTARGETS: command targets, that are in the commands/ subdir
CTARGETS = $(addprefix commands/, $(XTARGETS))


ifeq ($(DEBUG),1)
CPPFLAGS += -g
CFLAGS += -g
NVCCFLAGS += -g
endif

ifeq ($(UBSAN),1)
CFLAGS += -fsanitize=undefined,bounds-strict -fno-sanitize-recover=all
ifeq ($(DEBUG),0)
CFLAGS += -fsanitize-undefined-trap-on-error
endif
endif

ifeq ($(ASAN),1)
CFLAGS += -fsanitize=address
endif

ifeq ($(NOEXEC_STACK),1)
CPPFLAGS += -DNOEXEC_STACK
endif


ifeq ($(PARALLEL),1)
MAKEFLAGS += -j$(PARALLEL_NJOBS)
endif



CPPFLAGS += $(DEPFLAG) -iquote $(srcdir)/
CFLAGS += -std=gnu17
CXXFLAGS += -std=c++17




default: bart .gitignore


-include $(ALLDEPS)




# cuda

NVCC?=$(CUDA_BASE)/bin/nvcc


ifeq ($(CUDA),1)
CUDA_H := -I$(CUDA_BASE)/include
CPPFLAGS += -DUSE_CUDA $(CUDA_H)
ifeq ($(CUDNN),1)
CUDNN_H := -I$(CUDNN_BASE)/include
CPPFLAGS += -DUSE_CUDNN $(CUDNN_H)
endif
ifeq ($(BUILDTYPE), MacOSX)
CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -lcufft -lcudart -lcublas -m64 -lstdc++
else
ifeq ($(CUDNN),1)
CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -L$(CUDNN_BASE)/$(CUDNN_LIB) -lcudnn -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath $(CUDA_BASE)/$(CUDA_LIB) -Wl,-rpath $(CUDNN_BASE)/$(CUDNN_LIB)
else
CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath $(CUDA_BASE)/$(CUDA_LIB)
endif
endif
else
CUDA_H :=
CUDA_L :=
endif

# sm_20 no longer supported in CUDA 9
GPUARCH_FLAGS ?=
CUDA_CC ?= $(CC)
NVCCFLAGS += -DUSE_CUDA -Xcompiler -fPIC -O2 $(GPUARCH_FLAGS) -I$(srcdir)/ -m64 -ccbin $(CUDA_CC)
#NVCCFLAGS = -Xcompiler -fPIC -Xcompiler -fopenmp -O2  -I$(srcdir)/


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@
	$(NVCC) $(NVCCFLAGS) -M $^ -o $(DEPFILE)


# OpenMP

ifeq ($(OMP),1)
ifneq ($(BUILDTYPE), MacOSX)
CFLAGS += -fopenmp
CXXFLAGS += -fopenmp
NVCCFLAGS += -Xcompiler -fopenmp
else
ifeq ($(MACPORTS),1)
CFLAGS += -fopenmp
CXXFLAGS += -fopenmp
NVCCFLAGS += -Xcompiler -fopenmp
else
LDFLAGS += "-L/usr/local/opt/libomp/lib" -lomp
CPPFLAGS += "-I/usr/local/opt/libomp/include" -Xclang -fopenmp
endif
endif
else
CFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unknown-pragmas
endif

# Message Passing Interface
ifeq ($(MPI),1)
CFLAGS += -DUSE_MPI
CC = mpicc
endif

# BLAS/LAPACK
ifeq ($(SCALAPACK),1)
BLAS_L :=  -lopenblas -lscalapack
CPPFLAGS += -DUSE_OPENBLAS
CFLAGS += -DUSE_OPENBLAS
else
ifeq ($(ACML),1)
BLAS_H := -I$(ACML_BASE)/include
BLAS_L := -L$(ACML_BASE)/lib -lgfortran -lacml_mp -Wl,-rpath $(ACML_BASE)/lib
CPPFLAGS += -DUSE_ACML
else
ifeq ($(BUILDTYPE), MSYS)
BLAS_H := -I$(BLAS_BASE)
else
BLAS_H := -I$(BLAS_BASE)/include
endif
ifeq ($(BUILDTYPE), MacOSX)
BLAS_L := -L$(BLAS_BASE)/lib -lopenblas
else
ifeq ($(BUILDTYPE), MSYS)
	BLAS_L := -L/mingw64/lib -lopenblas
else
ifeq ($(BUILDTYPE), WASM)
	BLAS_L := -L$(BLAS_BASE)/lib
else
BLAS_L := -Wl,-rpath $(BLAS_BASE)/lib -L$(BLAS_BASE)/lib

ifeq ($(NOLAPACKE),1)
BLAS_L += -llapack -lblas
CPPFLAGS += -Isrc/lapacke
else
ifeq ($(OPENBLAS), 1)
ifeq ($(FORTRAN), 0)
BLAS_L += -lopenblas
else
BLAS_L += -llapacke -lopenblas
endif
CPPFLAGS += -DUSE_OPENBLAS
CFLAGS += -DUSE_OPENBLAS
else
BLAS_L += -llapacke -lblas
endif
endif
endif
endif
endif
endif
endif

ifeq ($(MKL),1)
BLAS_H := -I$(MKL_BASE)/include
BLAS_L := -L$(MKL_BASE)/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core
CPPFLAGS += -DUSE_MKL -DMKL_Complex8="complex float" -DMKL_Complex16="complex double"
CFLAGS += -DUSE_MKL -DMKL_Complex8="complex float" -DMKL_Complex16="complex double"
endif

ifeq ($(BLAS_THREADSAFE),1)
CPPFLAGS += -DBLAS_THREADSAFE
CFLAGS += -DBLAS_THREADSAFE
endif

ifeq ($(NON_DETERMINISTIC),1)
CPPFLAGS += -DNON_DETERMINISTIC
CFLAGS += -DNON_DETERMINISTIC
NVCCFLAGS += -DNON_DETERMINISTIC
endif


CPPFLAGS += $(FFTW_H) $(BLAS_H)

# librt
ifeq ($(BUILDTYPE), MacOSX)
	LIBRT :=
else
	LIBRT := -lrt
endif

# png
ifeq ($(PNG), 0)
PNG_L :=
CFLAGS += -DNO_PNG
CPPFLAGS += -DNO_PNG
else
PNG_L := -lpng
endif

ifeq ($(SLINK),1)
PNG_L += -lz
ifeq ($(DEBUG_DWARF),1)
LIBS += -lelf -lz -llzma -lbz2
endif
endif

ifeq ($(LINKER),icc)
PNG_L += -lz
endif



# fftw

FFTW_H := -I$(FFTW_BASE)/include/
ifeq ($(BUILDTYPE), WASM)
	FFTW_L :=  -L$(FFTW_BASE)/lib -lfftw3f
else
FFTW_L :=  -Wl,-rpath $(FFTW_BASE)/lib -L$(FFTW_BASE)/lib -lfftw3f
endif

ifeq ($(FFTWTHREADS),1)
ifneq ($(BUILDTYPE), MSYS)
	FFTW_L += -lfftw3f_threads
	CPPFLAGS += -DFFTWTHREADS
endif
endif



# ISMRM

ifeq ($(ISMRMRD),1)
ISMRM_H := -I$(ISMRM_BASE)/include
ISMRM_L := -L$(ISMRM_BASE)/lib -lismrmrd
ISMRM_H += -I /usr/include/hdf5/serial/
else
ISMRM_H :=
ISMRM_L :=
endif


# Logging backends

ifeq ($(LOG_BACKEND),1)
CPPFLAGS += -DUSE_LOG_BACKEND
ifeq ($(LOG_SIEMENS_BACKEND),1)
miscextracxxsrcs += $(srcdir)/misc/UTrace.cc
endif
ifeq ($(LOG_ORCHESTRA_BACKEND),1)
miscextracxxsrcs += $(srcdir)/misc/Orchestra.cc
endif
endif


ifeq ($(ISMRMRD),1)
miscextracxxsrcs += $(srcdir)/ismrm/xml_wrapper.cc
CPPFLAGS += $(ISMRM_H)
LIBS += -lstdc++
endif

ifeq ($(PYTORCH),1)
nnextracxxsrcs += $(srcdir)/nn/pytorch_cpp_wrapper.cc
LIBS += -lstdc++
endif


# change for static linking

ifeq ($(SLINK),1)
ifeq ($(SCALAPACK),1)
BLAS_L += -lgfortran -lquadmath
else
# work around fortran problems with static linking
LDFLAGS += -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition
ifneq ($(BUILDTYPE), MSYS)
LIBS += -lmvec
BLAS_L += -llapack -lblas
endif
BLAS_L += -lgfortran -lquadmath
endif
endif



# Modules

.LIBPATTERNS := lib%.a

vpath %.a lib
vpath % commands/

boxextrasrcs := $(XTARGETS:%=src/%.c)

define alib
$(1)srcs := $(wildcard $(srcdir)/$(1)/*.c)
$(1)cudasrcs := $(wildcard $(srcdir)/$(1)/*.cu)
$(1)objs := $$($(1)srcs:.c=.o)
$(1)winobjs := $$($(1)srcs:.c=.win.o)
$(1)objs += $$($(1)extrasrcs:.c=.o)
$(1)objs += $$($(1)extracxxsrcs:.cc=.o)

ifeq ($(CUDA),1)
$(1)objs += $$($(1)cudasrcs:.cu=.o)
endif

.INTERMEDIATE: $$($(1)objs)
.INTERMEDIATE: $$($(1)winobjs)

lib/lib$(1).a: lib$(1).a($$($(1)objs))

endef

ALIBS = misc num grecon sense noir iter linops wavelet lowrank noncart calib simu sake nlops moba lapacke box geom networks nn motion stl seq

ifeq ($(ISMRMRD),1)
ALIBS += ismrm
endif
ifeq ($(BUILDTYPE), MSYS)
ALIBS += win
endif
ifeq ($(BARTDLL), 1)
ALIBS += win
endif
$(eval $(foreach t,$(ALIBS),$(eval $(call alib,$(t)))))


# additional rules for lib misc
$(shell $(root)/rules/update_version.sh)
$(srcdir)/misc/version.o: $(srcdir)/misc/version.inc


# additional rules for lib ismrm
lib/libismrm.a: CPPFLAGS += $(ISMRM_H)

# additional rules for lib box
lib/libbox.a: CPPFLAGS += -include src/main.h

# lib calib
UTARGETS += test_grog test_casorati
MODULES_test_grog += -lcalib -lnoncart -lsimu -lgeom -lstl
MODULES_test_casorati+= -lcalib -llinops -liter


# shared libraries
define dlllib
$(1)srcs := $(wildcard $(srcdir)/$(1)/*.c)
$(1)objs := $$($(1)srcs:.c=.win.o)

.INTERMEDIATE: $$($(1)objs)

lib/$(1).dll: $$($(1)objs)

endef

DLLS=seq

$(eval $(foreach t,$(DLLS),$(eval $(call dlllib,$(t)))))



# lib linop
UTARGETS += test_linop_matrix test_linop test_padding
MODULES_test_linop += -llinops
MODULES_test_linop_matrix += -llinops
MODULES_test_padding += -llinops

# lib lowrank
UTARGETS += test_batchsvd
MODULES_test_batchsvd = -llowrank

# lib misc
UTARGETS += test_pattern test_types test_misc test_memcfl test_tree test_streams

# lib moba
UTARGETS += test_moba
MODULES_test_moba += -lmoba -lnoir -llowrank -lwavelet -liter -lnlops -llinops -lseq -lsimu

# lib nlop
UTARGETS += test_nlop test_nlop_jacobian
MODULES_test_nlop += -lnlops -lnoncart -llinops -liter
MODULES_test_nlop_jacobian += -lnlops -llinops

# lib noncart
UTARGETS += test_nufft test_fib
MODULES_test_nufft += -lnoncart -llinops
MODULES_test_fib += -lnoncart

# lib seq
UTARGETS += test_gradient test_events test_angle_calc test_adc_rf test_flash test_seq test_pulseq
UTARGETS_WINE += test_seq.win
MODULES_test_gradient += -lseq
MODULES_test_events += -lseq
MODULES_test_angle_calc += -lseq -lsimu -lnoncart
MODULES_test_adc_rf += -lseq -lsimu -lnoncart
MODULES_test_flash += -lseq -lsimu -lnoncart
MODULES_test_seq += -lseq -lsimu -lnoncart
MODULES_test_pulseq += -lseq -lnoncart

# lib num
UTARGETS += test_multind test_flpmath test_splines test_linalg test_polynom test_window test_conv
UTARGETS += test_ode test_nlmeans test_rand test_matexp
UTARGETS += test_blas test_mdfft test_ops test_ops_p test_flpmath2 test_convcorr test_specfun test_qform test_fft test_gaussians test_md_gaussians
UTARGETS += test_lapack
UTARGETS += test_morph
UTARGETS += test_linalg_rand
MODULES_test_linalg_rand += -llinops
ifeq ($(MPI),1)
UTARGETS += test_mpi test_mpi_multind test_mpi_flpmath test_mpi_fft
UTARGETS += test_mpi_reduction
endif
UTARGETS_GPU += test_cudafft test_cuda_flpmath test_cuda_flpmath2 test_cuda_gpukrnls test_cuda_convcorr test_cuda_multind test_cuda_shuffle test_cuda_memcache_clear test_cuda_rand

# lib simu
UTARGETS += test_ode_bloch test_ode_simu test_biot_savart test_signals test_epg test_pulse test_tsegf test_phantom
MODULES_test_ode_bloch += -lsimu
MODULES_test_ode_simu += -lseq -lsimu
MODULES_test_biot_savart += -lsimu
MODULES_test_signals += -lsimu
MODULES_test_epg += -lsimu
MODULES_test_pulse += -lseq -lsimu
MODULES_test_tsegf += -lsimu
MODULES_test_phantom += -lstl -lgeom -lsimu -lgeom

# lib geom
UTARGETS += test_geom test_stl
MODULES_test_geom += -lgeom
MODULES_test_stl += -lstl

# lib iter
UTARGETS += test_iter test_prox test_prox2 test_asl
MODULES_test_iter += -liter -lnlops -llinops
MODULES_test_prox += -liter -llinops
MODULES_test_prox2 += -liter -llinops -lnlops
MODULES_test_asl += -liter -llinops -lnlops

# lib nn
ifeq ($(TENSORFLOW),1)
UTARGETS += test_nn_tf
MODULES_test_nn_tf += -lnn -lnlops -llinops
endif

ifeq ($(PYTORCH),1)
UTARGETS += test_nn_pytorch
UTARGETS_GPU += test_nn_pytorch_cuda
MODULES_test_nn_pytorch += -lnn -lnlops -llinops
MODULES_test_nn_pytorch_cuda += -lnn -lnlops -llinops
endif


UTARGETS += test_nn_ops test_nn
MODULES_test_nn_ops += -lnn -lnlops -llinops -liter
MODULES_test_nn += -lnn -lnlops -llinops -liter

UTARGETS += test_affine
MODULES_test_affine+= -lmotion -lnlops -llinops -liter

UTARGETS_GPU += test_cuda_affine
MODULES_test_cuda_affine+= -lmotion -lnlops -llinops -liter

UTARGETS += test_score
MODULES_test_score += -lnetworks -lnn -lnlops -llinops -liter

UTARGETS += test_snlop
MODULES_test_snlop+= -lnlops -llinops


.gitignore: .gitignore.main Makefile*
	@echo '# AUTOGENERATED. DO NOT EDIT. (are you looking for .gitignore.main ?)' > .gitignore
	cat .gitignore.main >> .gitignore
	@echo /bart >> .gitignore
	@echo $(patsubst %, /%, $(CTARGETS) $(UTARGETS) $(UTARGETS_GPU)) | tr ' ' '\n' >> .gitignore


doc/commands.txt: bart
	./rules/update_commands.sh ./bart doc/commands.txt $(XTARGETS)

.PHONY: doxygen
doxygen: makedoc.sh doxyconfig bart
	 ./makedoc.sh


all: .gitignore $(CTARGETS) bart





# special targets

# always check if src/mainlist.inc exists:
-include .mainlist
.mainlist: src/mainlist.inc

src/mainlist.inc: $(ALLMAKEFILES)
	echo "#define MAIN_LIST $(XTARGETS:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_BASE $(TBASE:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_FLP $(TFLP:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_NUM $(TNUM:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_IO $(TIO:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_RECO $(TRECO:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_CALIB $(TCALIB:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_MRI $(TMRI:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_SIM $(TSIM:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_NN $(TNN:%=%,) ()" >> src/mainlist.inc
	echo "#define MAIN_MOTION $(TMOTION:%=%,) ()" >> src/mainlist.inc


$(CTARGETS): CPPFLAGS += -include src/main.h

bart: CPPFLAGS += -include src/main.h



LIBSEQ_NAME = bart_seq_$(shell git diff --quiet && git rev-parse --short=10 HEAD)


ifeq (32,$(ARCH))
MINGWDLLTOOL = i686-w64-mingw32-dlltool
else
MINGWDLLTOOL = x86_64-w64-mingw32-dlltool
endif

.PHONY: libseq_deploy
libseq_deploy: gitclean_check
	$(MAKE) lib/libbart.a
	$(MAKE) BARTDLL=1 bart.dll
	$(MINGWDLLTOOL) -l lib/$(LIBSEQ_NAME).lib --dllname $(LIBSEQ_NAME).dll -d bart.def
	cp lib/$(LIBSEQ_NAME).lib $(VM_BART_PATH)/lib/$(LIBSEQ_NAME).lib
	cp lib/libbart.a $(VM_BART_PATH)/lib/lib$(LIBSEQ_NAME).a
	cp bart.dll $(VM_BIN_PATH)/$(LIBSEQ_NAME).dll


.PHONY: gitclean_check
gitclean_check:
	git diff --quiet



# implicit rules

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@



# since GNU Make 4.4, archive members can be built separately from
# compilation, which means we do not need the special support for parallel
# building of archive members anymore
# see: https://www.gnu.org/software/make/manual/html_node/Archive-Pitfalls.html
ifeq ($(AR_LOCK_NEEDED),1)

# use for parallel make
AR=./ar_lock.sh
(%): %
	$(AR) $(ARFLAGS) $@ $%
else

# clear default archive member rule:
(%) : % ;
# when building .a files: run AR with all new .o files ($?)
%.a : ; $(AR) $(ARFLAGS) $@ $?
endif

ifeq (32,$(ARCH))
MINGWCC = i686-w64-mingw32-gcc
else
MINGWCC = x86_64-w64-mingw32-gcc
endif


%.win.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

# BLAS, LAPACK
WIN_NOT_SUPPORTED=%blas.win.o %lapack.win.o %blas_md_wrapper.win.o %vecops_strided.win.o %convcorr.win.o

bart.dll: CC = $(MINGWCC)
bart.dll: CFLAGS = -D NO_PNG -D NOLAPACKE -D NO_FFTW -D NO_LAPACK -D NO_BLAS -D NO_FIFO -D BARTDLL
bart.dll: CPPFLAGS = -D BARTLIB_EXPORTS -I$(srcdir)/
bart.dll: LDFLAGS = -shared -Wl,--subsystem,windows -Wl,--out-implib,bart.lib -Wl,--output-def,bart.def -static-libgcc
bart.dll: $(seqwinobjs) $(miscwinobjs) $(filter-out $(WIN_NOT_SUPPORTED),$(numwinobjs)) $(winwinobjs) $(noncartwinobjs) $(linopswinobjs) $(waveletwinobjs) $(geomwinobjs) $(stlwinobjs) $(simuwinobjs)
	$(CC) $^ $(LDFLAGS) -o $@

lib/libbart.a: CFLAGS = -D NO_PNG -D NOLAPACKE -D NO_FFTW -D NO_LAPACK -D NO_BLAS -D NO_FIFO -D BARTDLL -fPIC
lib/libbart.a: CPPFLAGS = -I$(srcdir)/
lib/libbart.a: $(seqobjs) $(miscobjs) $(filter-out $(WIN_NOT_SUPPORTED:.win.o=.o),$(numobjs)) $(winobjs) $(noncartobjs) $(linopsobjs) $(waveletobjs) $(geomobjs) $(stlobjs) $(simuobjs)
	$(AR) rcs $@ $^

$(UTARGETS_WINE): CC = $(MINGWCC)
$(UTARGETS_WINE): CPPFLAGS = -D BARTLIB_EXPORTS -I$(srcdir)/
$(UTARGETS_WINE): CFLAGS = -D NO_PNG -D NOLAPACKE -D NO_FFTW -D NO_LAPACK -D NO_BLAS -D NO_FIFO -D BARTDLL

.SECONDEXPANSION:
$(CTARGETS): commands/% : src/main.c $(srcdir)/%.o $$(MODULES_%) $(MODULES)
	$(LINKER) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$(@F) $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT) -o $@
ifeq ($(BUILDTYPE), WASM)
	./rules/add_node_shebang.sh $@
endif


.SECONDEXPANSION:
bart: % : src/main.c $(srcdir)/%.o $$(MODULES_%) $(MODULES)
ifeq ($(SHARED),1)
	$(LINKER) $(LDFLAGS) -shared $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT) -o bart.o
else
	$(LINKER) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$(@F) $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT) -o $@
endif
ifeq ($(BUILDTYPE), WASM)
	./rules/add_node_shebang.sh $@
endif



UTESTS=$(shell $(root)/utests/utests-collect.sh ./utests/$@.c)

.SECONDEXPANSION:
$(UTARGETS): % : utests/utest.c utests/%.o $$(MODULES_%) $(MODULES)
	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS="$(UTESTS)" $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm $(LIBRT) -o $@

UTESTS_GPU=$(shell $(root)/utests/utests_gpu-collect.sh ./utests/$@.c)

.SECONDEXPANSION:
$(UTARGETS_GPU): % : utests/utest.c utests/%.o $$(MODULES_%) $(MODULES)
	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS="$(UTESTS_GPU)" -DUTEST_GPU $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm $(LIBRT) -o $@

UTESTS_WINE=$(shell $(root)/utests/utests-collect.sh ./utests/$(@:.win=).c)

.SECONDEXPANSION:
$(UTARGETS_WINE): % : utests/utest.c utests/%.o bart.dll
	$(MINGWCC) $(CFLAGS) $(CPPFLAGS) -lbart -L. -DUTESTS="$(UTESTS_WINE)" -DUTEST_WINE -o $(@:.win=.exe) $+ -lm


# linker script version - does not work on MacOS X
#	$(CC) $(LDFLAGS) -Wl,-Tutests/utests.ld $(CFLAGS) -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) -lm -rt





# automatic tests

# system tests

ROOTDIR=$(root)
TOOLDIR=$(root)/commands
TESTS_DIR=$(root)/tests
TESTS_TMP_DIR?=$(TESTS_DIR)/tmp
TESTS_TMP=$(TESTS_TMP_DIR)/$$$$
TESTS_OUT=$(TESTS_DIR)/out


include $(root)/tests/*.mk

ifeq ($(BUILDTYPE), MSYS)
TMP_TESTS := $(TESTS)
NOT_SUPPORTED=tests/test-io tests/test-io2 tests/test-join-append tests/test-join-append-one tests/test-whiten
TESTS = $(filter-out $(NOT_SUPPORTED),$(TMP_TESTS))
endif


test:	${TESTS}

testslow: ${TESTS_SLOW}

testague: ${TESTS_AGUE} # test importing *.dat-files specified in tests/twixread.mk

gputest: ${TESTS_GPU}

mpitest: ${TESTS_MPI}

pythontest: ${TESTS_PYTHON}

ismrmrdtest: ${TESTS_ISMRMRD}



# unit tests

UTEST_RUN=

ifeq ($(MPI),1)
# only cfl files allowed with MPI
UTARGETS:=$(filter-out test_memcfl ,$(UTARGETS))
UTEST_RUN=mpirun -n 3
endif

ifeq ($(UTESTLEAK),1)
UTEST_RUN=valgrind --quiet --leak-check=full --error-exitcode=1 valgrind --log-file=/dev/null
endif

ifeq ($(BUILDTYPE), WASM)
UTEST_RUN=node
endif

.PHONY: utests-all utest utests_gpu-all utest_gpu

utests-all: $(UTARGETS)
	./utests/utests_run.sh "CPU" "$(UTEST_RUN)" $(UTARGETS)

utest: utests-all
	@echo ALL CPU UNIT TESTS PASSED.

utests_gpu-all: $(UTARGETS_GPU)
	./utests/utests_run.sh "GPU" "$(UTEST_RUN)" $(UTARGETS_GPU)

utest_gpu: utests_gpu-all
	@echo ALL GPU UNIT TESTS PASSED.

utest_wine-all: $(UTARGETS_WINE)
	./utests/utests_run.sh "WINE" "wine" $(UTARGETS_WINE:.win=.exe)

utest_wine: utest_wine-all
	@echo ALL WINE UNIT TESTS PASSED.


.PHONY: clean
clean:
	rm -f `find $(srcdir) -name "*.o"`
	rm -f $(root)/utests/*.o
	rm -f $(patsubst %, %, $(UTARGETS))
	rm -f $(patsubst %, %, $(UTARGETS_GPU))
	rm -f $(libdir)/.*.lock

.PHONY: allclean
allclean: clean
	rm -f $(libdir)/*.a $(ALLDEPS)
	rm -f $(root)/*.dll
	rm -f $(root)/*.lib
	rm -f $(root)/*.def
	rm -f bart
	rm -f $(patsubst commands/%, %, $(CTARGETS))
	rm -f $(CTARGETS)
	rm -f $(srcdir)/misc/version.inc
	rm -f $(srcdir)/mainlist.inc
	rm -rf $(root)/tests/tmp/*/
	rm -rf $(root)/stests/tmp/*/
	rm -rf $(root)/doc/dx
	rm -f $(root)/doc/commands.txt
	rm -f $(root)/save/fftw/*.fftw
	rm -f $(root)/save/nsv/*.dat
	touch isclean

.PHONY: distclean
distclean: allclean



-include isclean


isclean: $(ALLMAKEFILES)
ifeq ($(AUTOCLEAN),1)
	@echo "CONFIGURATION MODIFIED. RUNNING FULL REBUILD."
	touch isclean
	$(MAKE) allclean || rm isclean
else
ifneq ($(MAKECMDGOALS),allclean)
	@echo "CONFIGURATION MODIFIED."
endif
endif

# shared library
.PHONY: shared-lib
shared-lib:
	make allclean
	CFLAGS="-fPIC $(OPT) -Wmissing-prototypes" make
	gcc -shared -fopenmp src/bart.o -Wl,-whole-archive lib/lib*.a -Wl,-no-whole-archive -Wl,-Bdynamic $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm -lrt -o libbart.so
	make allclean

libbart.so: shared-lib


.PHONY: install
install: bart
	install -d $(DESTDIR)/$(PREFIX)/bin/
	install bart $(DESTDIR)/$(PREFIX)/bin/
	install -d $(DESTDIR)/$(PREFIX)/share/doc/bart/
	install $(root)/doc/*.txt $(root)/README $(DESTDIR)/$(PREFIX)/share/doc/bart/
	install -d $(DESTDIR)/$(PREFIX)/lib/bart/commands/


# generate release tar balls (identical to github)
%.tar.gz:
	git archive --prefix=bart-$(patsubst bart-%.tar.gz,%,$@)/ -o $@ v$(patsubst bart-%.tar.gz,%,$@)



# symbol table
bart.syms: bart
	rules/make_symbol_table.sh bart bart.syms



