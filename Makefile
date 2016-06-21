# Copyright 2013-2015. The Regents of the University of California.
# Copyright 2015-2016. Martin Uecker <martin.uecker@med.uni-goettingen.de>
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

CUDA?=0
ACML?=0
OMP?=1
SLINK?=0
DEBUG?=0
FFTWTHREADS?=1
ISMRMRD?=0

DESTDIR ?= /
PREFIX ?= usr/

BUILDTYPE = Linux
UNAME = $(shell uname -s)
NNAME = $(shell uname -n)

MYLINK=ln

ifeq ($(UNAME),Darwin)
	BUILDTYPE = MacOSX
	MYLINK = ln -s
endif

ifeq ($(BUILDTYPE), MacOSX)
	MACPORTS?=1
endif

ARFLAGS = r


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
	CC ?= gcc-mp-4.7
else
	CC ?= gcc
	# for symbols in backtraces
	LDFLAGS += -rdynamic
endif

CXX ?= g++



# openblas

ifneq ($(BUILDTYPE), MacOSX)
BLAS_BASE ?= /usr/
else
BLAS_BASE ?= /usr/local/opt/openblas/
ifeq ($(MACPORTS),1)
BLAS_BASE ?= /opt/local/
CPPFLAGS += -DUSE_MACPORTS
endif
endif

# cuda

CUDA_BASE ?= /usr/local/


# acml

ACML_BASE ?= /usr/local/acml/acml4.4.0/gfortran64_mp/

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

TBASE=show slice crop resize join transpose zeros ones flip circshift extract repmat bitmask reshape version
TFLP=scale conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg
TNUM=fft fftmod fftshift noise bench threshold conv rss filter
TRECO=pics pocsense rsense sqpics bpsense itsense nlinv nufft rof sake wave lrmatrix estdims
TCALIB=ecalib ecaltwo caldir walsh cc calmat svd estvar
TMRI=homodyne poisson twixread fakeksp
TSIM=phantom traj
TIO=toimg




MODULES = -lnum -lmisc -lnum -lmisc

MODULES_pics = -lgrecon -lsense -lwavelet2 -liter -llinops -lwavelet3 -llowrank -lnoncart
MODULES_sqpics = -lsense -lwavelet2 -liter -llinops -lwavelet3 -llowrank -lnoncart
MODULES_pocsense = -lsense -lwavelet2 -liter -llinops
MODULES_nlinv = -lnoir -liter
MODULES_rsense = -lgrecon -lsense -lnoir -lwavelet2 -lcalib -liter -llinops
MODULES_bpsense = -lsense -lwavelet2 -lnoncart -liter -llinops
MODULES_itsense = -liter -llinops
MODULES_ecalib = -lcalib
MODULES_ecaltwo = -lcalib
MODULES_caldir = -lcalib
MODULES_walsh = -lcalib
MODULES_calmat = -lcalib
MODULES_cc = -lcalib
MODULES_estvar = -lcalib
MODULES_nufft = -lnoncart -liter -llinops
MODULES_rof = -liter -llinops
MODULES_bench = -lwavelet2 -lwavelet3 -llinops
MODULES_phantom = -lsimu
MODULES_bart = -lbox -lgrecon -lsense -lnoir -lwavelet2 -liter -llinops -lwavelet3 -llowrank -lnoncart -lcalib -lsimu -lsake -ldfwavelet
MODULES_sake = -lsake
MODULES_wave = -liter -lwavelet2 -llinops -lsense
MODULES_threshold = -llowrank -lwavelet2 -liter -ldfwavelet -llinops
MODULES_fakeksp = -lsense -llinops
MODULES_lrmatrix = -llowrank -liter -llinops
MODULES_estdims = -lnoncart -llinops
MODULES_ismrmrd = -lismrm


-include Makefile.$(NNAME)
-include Makefile.local


ifeq ($(ISMRMRD),1)
TMRI += ismrmrd
MODULES_bart += -lismrm
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
default all clean allclean distclean doc/commands.txt doxygen test utest $(TARGETS):
	make MAKESTAGE=2 $(MAKECMDGOALS)
else


CPPFLAGS += $(DEPFLAG) -I$(srcdir)/
CFLAGS += -std=c99 -I$(srcdir)/
CXXFLAGS += -I$(srcdir)/




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

NVCCFLAGS = -DUSE_CUDA -Xcompiler -fPIC -Xcompiler -fopenmp -O3 -arch=sm_20 -I$(srcdir)/ -m64 -ccbin $(CC)
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
BLAS_L := -L$(BLAS_BASE)/lib -llapacke -lblas
endif
endif




CPPFLAGS += $(FFTW_H) $(BLAS_H)



# png
PNG_L := -lpng

ifeq ($(SLINK),1)
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

# change for static linking

ifeq ($(SLINK),1)
# work around fortran problems with static linking
LDFLAGS += -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -Wl,--allow-multiple-definition
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
	@echo AUTOGENERATED. DO NOT EDIT. > doc/commands.new
	for cmd in $(XTARGETS) ; do 		\
		printf "\n\n--%s--\n\n" $$cmd ;	\
		 ./bart $$cmd -h ;		\
	done >> doc/commands.new
	$(root)/rules/update-if-changed.sh doc/commands.new doc/commands.txt

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
	$(CC) $(LDFLAGS) $(CFLAGS) -Dmain_real=main_$@ -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) -lm
#	rm $(srcdir)/$@.o

UTESTS=$(shell $(root)/utests/utests-collect.sh ./utests/$@.c)

.SECONDEXPANSION:
$(UTARGETS): % : utests/utest.c utests/%.o $$(MODULES_%) $(MODULES)
	$(CC) $(LDFLAGS) $(CFLAGS) -DUTESTS="$(UTESTS)" -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) -lm


# linker script version - does not work on MacOS X
#	$(CC) $(LDFLAGS) -Wl,-Tutests/utests.ld $(CFLAGS) -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) -lm

clean:
	rm -f `find $(srcdir) -name "*.o"`
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


