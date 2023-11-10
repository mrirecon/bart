# Copyright 2013-2015. The Regents of the University of California.
# Copyright 2015-2022. Martin Uecker <uecker@tugraz.at>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.

# we have a two stage Makefile
MAKESTAGE ?= 1

# silent make
#MAKEFLAGS += --silent

# auto clean on makefile updates
AUTOCLEAN?=1

# clear out all implicit rules and variables
MAKEFLAGS += -R

# use for parallel make
AR=./ar_lock.sh

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
NOEXEC_STACK?=0
PARALLEL?=0
PARALLEL_NJOBS?=
FORTRAN?=1
PNG?=1
DEBUG_DWARF?=0

LOG_BACKEND?=0
LOG_SIEMENS_BACKEND?=0
LOG_ORCHESTRA_BACKEND?=0
LOG_GADGETRON_BACKEND?=0

# for debug backtraces
ifeq ($(DEBUG_DWARF),1)
LIBS += -ldw -lunwind
CPPFLAGS += -DUSE_DWARF
endif

DESTDIR ?= /
PREFIX ?= usr/local/

BUILDTYPE = Linux
UNAME = $(shell uname -s)
NNAME = $(shell uname -n)

MYLINK=ln


ifeq ($(UNAME),Darwin)
	BUILDTYPE = MacOSX
	MYLINK = ln -s
endif

ifeq ($(BUILDTYPE), MacOSX)
	MACPORTS ?= 1
endif


ifeq ($(BUILDTYPE), Linux)
	# as the defaults changed on most Linux distributions
	# explicitly specify non-deterministic archives to not break make
	ARFLAGS ?= rsU
else
	ARFLAGS ?= rs
endif

ifeq ($(BUILDTYPE), Linux)
ifneq (,$(findstring Red Hat,$(shell gcc --version)))
	CPPFLAGS+=-I/usr/include/lapacke/
	LDFLAGS+=-L/usr/lib64/atlas -ltatlas
endif
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


# Paths

here  = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
root := $(here)

srcdir = $(root)/src
libdir = $(root)/lib
bindir = $(root)/bin

export BART_TOOLBOX_PATH=$(root)


# Automatic dependency generation

DEPFILE = $(*D)/.$(*F).d
DEPFLAG = -MMD -MF $(DEPFILE)
ALLDEPS = $(shell find $(srcdir) utests -name ".*.d")


# Compilation flags

ifneq ($(DEBUG),1)
OPT = -O2
else
OPT = -Og
endif
#OPT += -ffp-contract=off
CPPFLAGS ?= -Wall -Wextra
CFLAGS ?= $(OPT) -Wmissing-prototypes
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

# cuda

CUDA_BASE ?= /usr/
CUDA_LIB ?= lib
CUDNN_BASE ?= $(CUDA_BASE)
CUDNN_LIB ?= lib64

# tensorflow
TENSORFLOW_BASE ?= /usr/local/

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


# Matlab

MATLAB_BASE ?= /usr/local/matlab/

# ISMRM

ISMRM_BASE ?= /usr/local/ismrmrd/



# Main build targets
#
TBASE=show slice crop resize join transpose squeeze flatten zeros ones flip circshift extract repmat bitmask reshape version delta copy casorati vec poly index multicfl
TFLP=scale invert conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg cabs zexp calc
TNUM=fft fftmod fftshift noise bench threshold conv rss filter nlmeans mandelbrot wavelet window var std fftrot roistat pol2mask conway morphop
TRECO=pics pocsense sqpics itsense nlinv moba nufft nufftbase rof tgv ictv sake wave lrmatrix estdims estshift estdelay wavepsf wshfl rtnlinv mobafit
TCALIB=ecalib ecaltwo caldir walsh cc ccapply calmat svd estvar whiten rmfreq ssa bin psf
TMRI=homodyne poisson twixread fakeksp looklocker upat fovshift
TSIM=phantom traj signal epg sim
TIO=toimg
TNN=reconet nnet onehotenc measure mnist tensorflow



MODULES = -lnum -lmisc -lnum -lmisc
ifeq ($(BUILDTYPE), MSYS)
MODULES += -lwin
endif

MODULES_pics = -lgrecon -lsense -liter -llinops -lwavelet -llowrank -lnoncart -lnn -lnlops 
MODULES_sqpics = -lsense -liter -llinops -lwavelet -llowrank -lnoncart -llinops
MODULES_pocsense = -lsense -liter -llinops -lwavelet
MODULES_nlinv = -lnoir -liter -lnlops -llinops -lnoncart
MODULES_rtnlinv = -lnoir -liter -lnlops -llinops -lnoncart
MODULES_moba = -lmoba -lnoir -lnn -lnlops -llinops -lwavelet -lnoncart -lsimu -lgrecon -llowrank -llinops -liter -lnn
MODULES_mobafit = -lmoba -lnlops -llinops -lsimu -liter -lnoir
MODULES_bpsense = -lsense -lnoncart -liter -llinops -lwavelet
MODULES_itsense = -liter -llinops
MODULES_ecalib = -lcalib -llinops
MODULES_ecaltwo = -lcalib -llinops
MODULES_estdelay = -lcalib
MODULES_caldir = -lcalib
MODULES_walsh = -lcalib
MODULES_calmat = -lcalib
MODULES_cc = -lcalib -llinops
MODULES_ccapply = -lcalib -llinops
MODULES_estvar = -lcalib
MODULES_nufft = -lnoncart -liter -llinops
MODULES_rof = -liter -llinops
MODULES_tgv = -liter -llinops
MODULES_ictv = -liter -llinops
MODULES_bench = -lwavelet -llinops
MODULES_phantom = -lsimu -lgeom
MODULES_bart = -lbox -lgrecon -lsense -lnoir -liter -llinops -lwavelet -llowrank -lnoncart -lcalib -lsimu -lsake -ldfwavelet -lnlops -lnetworks -lnn -liter -lmoba -lgeom -lnn  -lnlops
MODULES_sake = -lsake
MODULES_traj = -lnoncart
MODULES_wave = -liter -lwavelet -llinops -llowrank
MODULES_threshold = -llowrank -liter -ldfwavelet -llinops -lwavelet
MODULES_fakeksp = -lsense -llinops
MODULES_lrmatrix = -llowrank -liter -llinops -lnlops
MODULES_estdims =
MODULES_ismrmrd = -lismrm
MODULES_wavelet = -llinops -lwavelet
MODULES_wshfl = -lgrecon -lsense -liter -llinops -lwavelet -llowrank -lnoncart -lnlops -lnn -lnlops
MODULES_ssa = -lcalib
MODULES_bin = -lcalib
MODULES_signal = -lsimu
MODULES_pol2mask = -lgeom
MODULES_epg = -lsimu
MODULES_reconet = -lgrecon -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_mnist = -lnetworks -lnn -lnlops -llinops -liter
MODULES_nnet = -lgrecon -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_tensorflow = -lnn -lnlops -llinops -liter
MODULES_measure = -lgrecon -lnetworks -lnoncart -lnn -lnlops -llinops -liter
MODULES_onehotenc = -lnn
MODULES_sim = -lsimu
MODULES_morphop = -lnlops -llinops -lgeom
MODULES_psf = -lnoncart -llinops

MAKEFILES = $(wildcard $(root)/Makefiles/Makefile.*)
ALLMAKEFILES = $(root)/Makefile $(wildcard $(root)/Makefile.* $(root)/*.mk $(root)/rules/*.mk $(root)/Makefiles/Makefile.*)

-include Makefile.$(NNAME)
-include Makefile.local
-include $(MAKEFILES)


GCCVERSION11 := $(shell expr `$(CC) -dumpversion | cut -f1 -d.` \>= 11)
GCCVERSION14 := $(shell expr `$(CC) -dumpversion | cut -f1 -d.` \>= 14)

# clang

ifeq ($(findstring clang,$(CC)),clang)
	CFLAGS += -fblocks
	LDFLAGS += -lBlocksRuntime
else
# only add if not clang, as it doesn't understand this:
ifeq ($(GCCVERSION14), 1)
    CFLAGS += -Wuseless-cast
else
ifeq ($(GCCVERSION11), 1)
    CFLAGS += -Wno-vla-parameter -Wno-nonnull -Wno-maybe-uninitialized
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



XTARGETS += $(TBASE) $(TFLP) $(TNUM) $(TIO) $(TRECO) $(TCALIB) $(TMRI) $(TSIM) $(TNN)
# CTARGETS: command targets, that are in the commands/ subdir
CTARGETS = $(patsubst %, commands/%, $(XTARGETS))


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


ifeq ($(MAKESTAGE),1)
.PHONY: doc/commands.txt bart $(CTARGETS)
default all clean allclean distclean doc/commands.txt doxygen test utest utest_gpu gputest testslow pythontest testague scripttest shared-lib bart $(CTARGETS):
	$(MAKE) MAKESTAGE=2 $(MAKECMDGOALS)

tests/test-%: force
	$(MAKE) MAKESTAGE=2 $(MAKECMDGOALS)

force: ;

else


CPPFLAGS += $(DEPFLAG) -iquote $(srcdir)/
CFLAGS += -std=gnu11
CXXFLAGS += -std=c++14




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
CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -L$(CUDNN_BASE)/$(CUDNN_LIB) -lcudnn -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath $(CUDA_BASE)/$(CUDA_LIB)
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
NVCCFLAGS += -DUSE_CUDA -Xcompiler -fPIC -O2 $(GPUARCH_FLAGS) -I$(srcdir)/ -m64 -ccbin $(CC)
#NVCCFLAGS = -Xcompiler -fPIC -Xcompiler -fopenmp -O2  -I$(srcdir)/


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@
	$(NVCC) $(NVCCFLAGS) -M $^ -o $(DEPFILE)


# OpenMP

ifeq ($(OMP),1)
CFLAGS += -fopenmp
CXXFLAGS += -fopenmp
NVCCFLAGS += -Xcompiler -fopenmp
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
ifeq ($(NOLAPACKE),1)
BLAS_L := -L$(BLAS_BASE)/lib -llapack -lblas
CPPFLAGS += -Isrc/lapacke
else
ifeq ($(OPENBLAS), 1)
ifeq ($(FORTRAN), 0)
BLAS_L := -L$(BLAS_BASE)/lib -lopenblas
else
BLAS_L := -L$(BLAS_BASE)/lib -llapacke -lopenblas
endif
CPPFLAGS += -DUSE_OPENBLAS
CFLAGS += -DUSE_OPENBLAS
else
BLAS_L := -L$(BLAS_BASE)/lib -llapacke -lblas
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
FFTW_L := -L$(FFTW_BASE)/lib -lfftw3f

ifeq ($(FFTWTHREADS),1)
ifneq ($(BUILDTYPE), MSYS)
	FFTW_L += -lfftw3f_threads
	CPPFLAGS += -DFFTWTHREADS
endif
endif



# Matlab

MATLAB_H := -I$(MATLAB_BASE)/extern/include
MATLAB_L := -Wl,-rpath $(MATLAB_BASE)/bin/glnxa64 -L$(MATLAB_BASE)/bin/glnxa64 -lmat -lmx -lm -lstdc++

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
$(1)objs += $$($(1)extrasrcs:.c=.o)
$(1)objs += $$($(1)extracxxsrcs:.cc=.o)

ifeq ($(CUDA),1)
$(1)objs += $$($(1)cudasrcs:.cu=.o)
endif

.INTERMEDIATE: $$($(1)objs)

lib/lib$(1).a: lib$(1).a($$($(1)objs))

endef

ALIBS = misc num grecon sense noir iter linops wavelet lowrank noncart calib simu sake dfwavelet nlops moba lapacke box geom nn networks

ifeq ($(ISMRMRD),1)
ALIBS += ismrm
endif
ifeq ($(BUILDTYPE), MSYS)
ALIBS += win
endif
$(eval $(foreach t,$(ALIBS),$(eval $(call alib,$(t)))))


# additional rules for lib misc
$(eval $(shell $(root)/rules/update-version.sh))

$(srcdir)/misc/version.o: $(srcdir)/misc/version.inc


# additional rules for lib ismrm
lib/libismrm.a: CPPFLAGS += $(ISMRM_H)


# lib linop
UTARGETS += test_linop_matrix test_linop test_padding
MODULES_test_linop += -llinops
MODULES_test_linop_matrix += -llinops
MODULES_test_padding += -llinops

# lib lowrank
UTARGETS += test_batchsvd
MODULES_test_batchsvd = -llowrank

# lib misc
UTARGETS += test_pattern test_types test_misc test_memcfl test_tree

# lib moba
UTARGETS += test_moba
MODULES_test_moba += -lmoba -lnoir -llowrank -lwavelet -liter -lnlops -llinops -lsimu

# lib nlop
UTARGETS += test_nlop test_nlop_jacobian
MODULES_test_nlop += -lnlops -lnoncart -llinops -liter
MODULES_test_nlop_jacobian += -lnlops -llinops

# lib noncart
UTARGETS += test_nufft test_fib
MODULES_test_nufft += -lnoncart -llinops
MODULES_test_fib += -lnoncart

# lib num
UTARGETS += test_multind test_flpmath test_splines test_linalg test_polynom test_window test_conv test_ode test_nlmeans test_rand
UTARGETS += test_blas test_mdfft test_ops test_ops_p test_flpmath2 test_convcorr test_specfun test_qform test_fft test_gaussians
ifeq ($(MPI),1)
UTARGETS += test_mpi test_mpi_multind test_mpi_flpmath test_mpi_fft
endif
UTARGETS_GPU += test_cudafft test_cuda_flpmath test_cuda_flpmath2 test_cuda_gpukrnls test_cuda_convcorr test_cuda_multind test_cuda_shuffle test_cuda_memcache_clear

# lib simu
UTARGETS += test_ode_bloch test_ode_simu test_biot_savart test_signals test_epg test_pulse
MODULES_test_ode_bloch += -lsimu
MODULES_test_ode_simu += -lsimu
MODULES_test_biot_savart += -lsimu
MODULES_test_signals += -lsimu
MODULES_test_epg += -lsimu
MODULES_test_pulse += -lsimu

# lib geom
UTARGETS += test_geom
MODULES_test_geom += -lgeom

# lib iter
UTARGETS += test_iter test_prox test_prox2
MODULES_test_iter += -liter -lnlops -llinops
MODULES_test_prox += -liter -llinops
MODULES_test_prox2 += -liter -llinops -lnlops

# lib nn
ifeq ($(TENSORFLOW),1)
UTARGETS += test_nn_tf
MODULES_test_nn_tf += -lnn -lnlops -llinops
endif


UTARGETS += test_nn_ops test_nn
MODULES_test_nn_ops += -lnn -lnlops -llinops -liter
MODULES_test_nn += -lnn -lnlops -llinops -liter




XTARGETS:=$(sort $(XTARGETS))



.gitignore: .gitignore.main Makefile*
	@echo '# AUTOGENERATED. DO NOT EDIT. (are you looking for .gitignore.main ?)' > .gitignore
	cat .gitignore.main >> .gitignore
	@echo /bart >> .gitignore
	@echo $(patsubst %, /%, $(CTARGETS) $(UTARGETS) $(UTARGETS_GPU)) | tr ' ' '\n' >> .gitignore


doc/commands.txt: bart
	./rules/update_commands.sh ./bart doc/commands.txt $(XTARGETS)

doxygen: makedoc.sh doxyconfig bart
	 ./makedoc.sh


all: .gitignore bart $(CTARGETS)





# special targets


$(CTARGETS): CPPFLAGS += -DMAIN_LIST="$(XTARGETS:%=%,) ()" -include src/main.h


bart: CPPFLAGS += -DMAIN_LIST="$(XTARGETS:%=%,) ()" -include src/main.h


mat2cfl: $(srcdir)/mat2cfl.c -lnum -lmisc
	$(CC) $(CFLAGS) $(MATLAB_H) -omat2cfl  $+ $(MATLAB_L) $(CUDA_L)





# implicit rules

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

ifeq ($(PARALLEL),1)
(%): %
	$(AR) $(ARFLAGS) $@ $%
else
(%): %
	$(AR) $(ARFLAGS) $@ $%
endif




.SECONDEXPANSION:
$(CTARGETS): commands/% : src/main.c $(srcdir)/%.o $$(MODULES_%) $(MODULES)
	$(LINKER) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$(@F) -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT)


.SECONDEXPANSION:
bart: % : src/main.c $(srcdir)/%.o $$(MODULES_%) $(MODULES)
ifeq ($(SHARED),1)
	$(LINKER) $(LDFLAGS) -shared $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$@ -o bart.o $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT)
else
	$(LINKER) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$(@F) -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT)
endif

#	rm $(srcdir)/$@.o

UTESTS=$(shell $(root)/utests/utests-collect.sh ./utests/$@.c)

.SECONDEXPANSION:
$(UTARGETS): % : utests/utest.c utests/%.o $$(MODULES_%) $(MODULES)
	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS="$(UTESTS)" -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm $(LIBRT)

UTESTS_GPU=$(shell $(root)/utests/utests_gpu-collect.sh ./utests/$@.c)

.SECONDEXPANSION:
$(UTARGETS_GPU): % : utests/utest_gpu.c utests/%.o $$(MODULES_%) $(MODULES)
	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS_GPU="$(UTESTS_GPU)" -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm $(LIBRT)



# linker script version - does not work on MacOS X
#	$(CC) $(LDFLAGS) -Wl,-Tutests/utests.ld $(CFLAGS) -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) -lm -rt

clean:
	rm -f `find $(srcdir) -name "*.o"`
	rm -f $(root)/utests/*.o
	rm -f $(patsubst %, %, $(UTARGETS))
	rm -f $(patsubst %, %, $(UTARGETS_GPU))
	rm -f $(libdir)/.*.lock

allclean: clean
	rm -f $(libdir)/*.a $(ALLDEPS)
	rm -f bart
	rm -f $(patsubst %, %, $(CTARGETS))
	rm -f $(srcdir)/misc/version.inc
	rm -rf $(root)/tests/tmp/*/
	rm -rf $(root)/stests/tmp/*/
	rm -rf $(root)/doc/dx
	rm -f $(root)/doc/commands.txt
	rm -f $(root)/save/fftw/*.fftw
	rm -f $(root)/save/nsv/*.dat
	touch isclean

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



# automatic tests

# system tests

ROOTDIR=$(root)
TOOLDIR=$(root)/commands
TESTS_DIR=$(root)/tests
TESTS_TMP=$(TESTS_DIR)/tmp/$$$$
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

pythontest: ${TESTS_PYTHON}


# unit tests

UTEST_RUN=

ifeq ($(MPI),1)
# only cfl files allowed with MPI
UTARGETS:=$(filter-out test_memcfl ,$(UTARGETS))
UTEST_RUN=mpirun --allow-run-as-root -n 3
endif

ifeq ($(UTESTLEAK),1)
# we blacklist some targets because valgrind crashes (blas related)
UTARGETS:=$(filter-out test_flpmath test_blas,$(UTARGETS))
UTEST_RUN=valgrind --quiet --leak-check=full --error-exitcode=1 valgrind --suppressions=./valgrind.supp --log-file=/dev/null
endif

# define space to faciliate running executables
define \n


endef

utests-all: $(UTARGETS)
	$(patsubst %,$(\n)$(UTEST_RUN) ./%,$(UTARGETS))

utest: utests-all
	@echo ALL CPU UNIT TESTS PASSED.

utests_gpu-all: $(UTARGETS_GPU)
	$(patsubst %,$(\n)./%,$(UTARGETS_GPU))

utest_gpu: utests_gpu-all
	@echo ALL GPU UNIT TESTS PASSED.

# shared library
shared-lib:
	make allclean
	CFLAGS="-fPIC $(OPT) -Wmissing-prototypes" make
	gcc -shared -fopenmp -o libbart.so src/bart.o -Wl,-whole-archive lib/lib*.a -Wl,-no-whole-archive -Wl,-Bdynamic $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm -lrt
	make allclean

endif	# MAKESTAGE


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



