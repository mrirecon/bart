### Custom Makefiles
Put custom Makefiles here, to be included in the standard Makefile.
The build will automatically include the following files in this directory
matching the expansion `Makefiles.*`

Example custom Makefile for modifying build:

```bash
## Makefile.local
# Makefile for my local build

DEBUG = 1

# Parallel make
PARALLEL ?= 1

# GPU
CUDA=0

CC=clang
OMP=0

# Paths
FFTW_BASE := /opt/local/
MATLAB_BASE := /Applications/MATLAB_R2016a.app
CUDA_BASE = /usr/local/cuda/
BLAS_BASE := /opt/local
```

Example Makefile and library rules for adding a custom program:

```bash
## Makefiles/Makefile.sum
# Compile my custom program, src/sum.c, which relies on
# my custom library, lib/libsum.a

MODULES_sum = -lsum
MODULES_bart += -lsum
XTARGETS += sum
```

```bash
### rules/sum.mk
# Build my custom library with files under src/sum/

sumsrcs := $(wildcard $(srcdir)/sum/*.c)
sumobjs := $(sumsrcs:.c=.o)

.INTERMEDIATE: $(sumobjs)

lib/libsum.a: libsum.a($(sumobjs))
```
