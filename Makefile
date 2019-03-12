# Copyright 2013-2015. The Regents of the University of California.
# Copyright 2015-2018. Martin Uecker <martin.uecker@med.uni-goettingen.de>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.

# we have a two stage Makefile
MAKESTAGE ?= 1

# silent make
#MAKEFLAGS += --silent

# clear out all implicit rules and variables
MAKEFLAGS += -R

# use for parallel make
AR=./ar_lock.sh

MKL?=0
CUDA?=0
ACML?=0
OMP?=1
SLINK?=0
DEBUG?=0
FFTWTHREADS?=1
ISMRMRD?=0

LOG_BACKEND?=0
LOG_SIEMENS_BACKEND?=0
LOG_ORCHESTRA_BACKEND?=0
LOG_GADGETRON_BACKEND?=0
ENABLE_MEM_CFL?=0
MEMONLY_CFL?=0


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


ifeq ($(UNAME),Cygwin)
	BUILDTYPE = Cygwin
	NOLAPACKE ?= 1
endif

ifeq ($(UNAME),CYGWIN_NT-10.0)
	BUILDTYPE = Cygwin
	NOLAPACKE ?= 1
endif




# Paths

here  = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
root := $(here)

srcdir = $(root)/src
libdir = $(root)/lib
bindir = $(root)/bin

export TOOLBOX_PATH=$(root)


# Automatic dependency generation

DEPFILE = $(*D)/.$(*F).d
DEPFLAG = -MMD -MF $(DEPFILE)
ALLDEPS = $(shell find $(srcdir) utests -name ".*.d")


# Compilation flags

OPT = -O3 -ffast-math
CPPFLAGS ?= -Wall -Wextra
CFLAGS ?= $(OPT) -Wmissing-prototypes
CXXFLAGS ?= $(OPT)

ifeq ($(BUILDTYPE), MacOSX)
	CC ?= gcc-mp-6
else
	CC ?= gcc
	# for symbols in backtraces
	LDFLAGS += -rdynamic
endif




# openblas

ifneq ($(BUILDTYPE), MacOSX)
BLAS_BASE ?= /usr/
else
ifeq ($(MACPORTS),1)
BLAS_BASE ?= /opt/local/
CPPFLAGS += -DUSE_MACPORTS
endif
BLAS_BASE ?= /usr/local/opt/openblas/
endif

# cuda

CUDA_BASE ?= /usr/local/


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



# Main build targets are defined in build_targets.mk so that both CMake and Make can use the same definitions
# set values for TBASE TFLP TNUM TRECO TCALIB TMRI TSIM TIO in build_targets.mk
include build_targets.mk



MODULES = -lnum -lmisc -lnum -lmisc

MODULES_pics = -lgrecon -lsense -liter -llinops -lwavelet -llowrank -lnoncart
MODULES_sqpics = -lsense -liter -llinops -lwavelet -llowrank -lnoncart
MODULES_pocsense = -lsense -liter -llinops -lwavelet
MODULES_nlinv = -lnoir -liter -lnlops -llinops
MODULES_bpsense = -lsense -lnoncart -liter -llinops -lwavelet
MODULES_itsense = -liter -llinops
MODULES_ecalib = -lcalib
MODULES_ecaltwo = -lcalib
MODULES_caldir = -lcalib
MODULES_walsh = -lcalib
MODULES_calmat = -lcalib
MODULES_cc = -lcalib
MODULES_ccapply = -lcalib
MODULES_estvar = -lcalib
MODULES_nufft = -lnoncart -liter -llinops
MODULES_rof = -liter -llinops
MODULES_bench = -lwavelet -llinops
MODULES_phantom = -lsimu
MODULES_bart = -lbox -lgrecon -lsense -lnoir -liter -llinops -lwavelet -llowrank -lnoncart -lcalib -lsimu -lsake -ldfwavelet -lnlops
MODULES_sake = -lsake
MODULES_wave = -liter -lwavelet -llinops -llowrank
MODULES_threshold = -llowrank -liter -ldfwavelet -llinops -lwavelet
MODULES_fakeksp = -lsense -llinops
MODULES_lrmatrix = -llowrank -liter -llinops
MODULES_estdims = -lnoncart -llinops
MODULES_ismrmrd = -lismrm
MODULES_wavelet = -llinops -lwavelet
MODULES_wshfl = -llinops -lwavelet -liter -llowrank


MAKEFILES = $(root)/Makefiles/Makefile.*

-include Makefile.$(NNAME)
-include Makefile.local
-include $(MAKEFILES)


# clang

ifeq ($(findstring clang,$(CC)),clang)
	CFLAGS += -fblocks
	LDFLAGS += -lBlocksRuntime
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



XTARGETS += $(TBASE) $(TFLP) $(TNUM) $(TIO) $(TRECO) $(TCALIB) $(TMRI) $(TSIM)
TARGETS = bart $(XTARGETS)



ifeq ($(DEBUG),1)
CPPFLAGS += -g
CFLAGS += -g
endif


ifeq ($(PARALLEL),1)
MAKEFLAGS += -j
endif


ifeq ($(MAKESTAGE),1)
.PHONY: doc/commands.txt $(TARGETS)
default all clean allclean distclean doc/commands.txt doxygen test utest gputest $(TARGETS):
	make MAKESTAGE=2 $(MAKECMDGOALS)
else


CPPFLAGS += $(DEPFLAG) -iquote $(srcdir)/
CFLAGS += -std=gnu11
CXXFLAGS += -std=c++11




default: bart doc/commands.txt .gitignore


-include $(ALLDEPS)




# cuda

NVCC = $(CUDA_BASE)/bin/nvcc


ifeq ($(CUDA),1)
CUDA_H := -I$(CUDA_BASE)/include
CPPFLAGS += -DUSE_CUDA $(CUDA_H)
ifeq ($(BUILDTYPE), MacOSX)
CUDA_L := -L$(CUDA_BASE)/lib -lcufft -lcudart -lcublas -m64 -lstdc++
else
CUDA_L := -L$(CUDA_BASE)/lib64 -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath $(CUDA_BASE)/lib64
endif 
else
CUDA_H :=
CUDA_L :=  
endif

# sm_20 no longer supported in CUDA 9
GPUARCH_FLAGS ?= 
NVCCFLAGS = -DUSE_CUDA -Xcompiler -fPIC -Xcompiler -fopenmp -O3 $(GPUARCH_FLAGS) -I$(srcdir)/ -m64 -ccbin $(CC)
#NVCCFLAGS = -Xcompiler -fPIC -Xcompiler -fopenmp -O3  -I$(srcdir)/


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@
	$(NVCC) $(NVCCFLAGS) -M $^ -o $(DEPFILE)


# OpenMP

ifeq ($(OMP),1)
CFLAGS += -fopenmp
CXXFLAGS += -fopenmp
else
CFLAGS += -Wno-unknown-pragmas
CXXFLAGS += -Wno-unknown-pragmas
endif



# BLAS/LAPACK

ifeq ($(ACML),1)
BLAS_H := -I$(ACML_BASE)/include
BLAS_L := -L$(ACML_BASE)/lib -lgfortran -lacml_mp -Wl,-rpath $(ACML_BASE)/lib
CPPFLAGS += -DUSE_ACML
else
BLAS_H := -I$(BLAS_BASE)/include
ifeq ($(BUILDTYPE), MacOSX)
BLAS_L := -L$(BLAS_BASE)/lib -lopenblas
else
ifeq ($(NOLAPACKE),1)
BLAS_L := -L$(BLAS_BASE)/lib -llapack -lblas
CPPFLAGS += -Isrc/lapacke
else
BLAS_L := -L$(BLAS_BASE)/lib -llapacke -lblas
endif
endif
endif

ifeq ($(MKL),1)
BLAS_H := -I$(MKL_BASE)/include
BLAS_L := -L$(MKL_BASE)/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core
CPPFLAGS += -DUSE_MKL -DMKL_Complex8="complex float" -DMKL_Complex16="complex double"
CFLAGS += -DUSE_MKL -DMKL_Complex8="complex float" -DMKL_Complex16="complex double"
endif



CPPFLAGS += $(FFTW_H) $(BLAS_H)



# png
PNG_L := -lpng

ifeq ($(SLINK),1)
	PNG_L += -lz
endif

ifeq ($(LINKER),icc)
	PNG_L += -lz
endif



# fftw

FFTW_H := -I$(FFTW_BASE)/include/
FFTW_L := -L$(FFTW_BASE)/lib -lfftw3f

ifeq ($(FFTWTHREADS),1)
	FFTW_L += -lfftw3f_threads
	CPPFLAGS += -DFFTWTHREADS
endif

# Matlab

MATLAB_H := -I$(MATLAB_BASE)/extern/include
MATLAB_L := -Wl,-rpath $(MATLAB_BASE)/bin/glnxa64 -L$(MATLAB_BASE)/bin/glnxa64 -lmat -lmx -lm -lstdc++

# ISMRM

ifeq ($(ISMRMRD),1)
ISMRM_H := -I$(ISMRM_BASE)/include
ISMRM_L := -L$(ISMRM_BASE)/lib -lismrmrd
else
ISMRM_H :=
ISMRM_L :=
endif

# Enable in-memory CFL files

ifeq ($(ENABLE_MEM_CFL),1)
CPPFLAGS += -DUSE_MEM_CFL
miscextracxxsrcs += $(srcdir)/misc/mmiocc.cc
LDFLAGS += -lstdc++
endif

# Only allow in-memory CFL files (ie. disable support for all other files)

ifeq ($(MEMONLY_CFL),1)
CPPFLAGS += -DMEMONLY_CFL
miscextracxxsrcs += $(srcdir)/misc/mmiocc.cc
LDFLAGS += -lstdc++
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


# change for static linking

ifeq ($(SLINK),1)
# work around fortran problems with static linking
LDFLAGS += -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition
LIBS += -lmvec
BLAS_L += -llapack -lblas -lgfortran -lquadmath
endif



# Modules

.LIBPATTERNS := lib%.a


vpath %.a lib

DIRS = $(root)/rules/*.mk

include $(DIRS)

# sort BTARGETS after everything is included
BTARGETS:=$(sort $(BTARGETS))
XTARGETS:=$(sort $(XTARGETS))



.gitignore: .gitignore.main Makefile*
	@echo '# AUTOGENERATED. DO NOT EDIT. (are you looking for .gitignore.main ?)' > .gitignore
	cat .gitignore.main >> .gitignore
	@echo $(patsubst %, /%, $(TARGETS) $(UTARGETS)) | tr ' ' '\n' >> .gitignore


doc/commands.txt: bart
	./rules/update_commands.sh ./bart doc/commands.txt $(XTARGETS)

doxygen: makedoc.sh doxyconfig bart
	 ./makedoc.sh


all: .gitignore $(TARGETS)





# special targets


$(XTARGETS): CPPFLAGS += -DMAIN_LIST="$(XTARGETS:%=%,) ()" -include src/main.h


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
	rm $%
endif


# we add the rm because intermediate files are not deleted
# automatically for some reason
# (but it produces errors for parallel builds for make all)



.SECONDEXPANSION:
$(TARGETS): % : src/main.c $(srcdir)/%.o $$(MODULES_%) $(MODULES)
	$(LINKER) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$@ -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm
#	rm $(srcdir)/$@.o

UTESTS=$(shell $(root)/utests/utests-collect.sh ./utests/$@.c)

.SECONDEXPANSION:
$(UTARGETS): % : utests/utest.c utests/%.o $$(MODULES_%) $(MODULES)
	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS="$(UTESTS)" -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm


# linker script version - does not work on MacOS X
#	$(CC) $(LDFLAGS) -Wl,-Tutests/utests.ld $(CFLAGS) -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) -lm

clean:
	rm -f `find $(srcdir) -name "*.o"`
	rm -f utests/*.o
	rm -f $(patsubst %, %, $(UTARGETS))
	rm -f $(root)/lib/.*.lock

allclean: clean
	rm -f $(libdir)/*.a $(ALLDEPS)
	rm -f $(patsubst %, %, $(TARGETS))
	rm -f $(srcdir)/misc/version.inc
	rm -rf doc/dx
	rm -f doc/commands.txt

distclean: allclean


# automatic tests

# system tests

TOOLDIR=$(root)
TESTS_TMP=$(root)/tests/tmp/$$$$/
TESTS_OUT=$(root)/tests/out/


include $(root)/tests/*.mk

test:	${TESTS}

gputest: ${TESTS_GPU}

pythontest: ${TESTS_PYTHON}

# unit tests

# define space to faciliate running executables
define \n


endef

utests-all: $(UTARGETS)
	$(patsubst %,$(\n)./%,$(UTARGETS))

utest: utests-all
	@echo ALL UNIT TESTS PASSED.


endif	# MAKESTAGE


install: bart $(root)/doc/commands.txt
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

