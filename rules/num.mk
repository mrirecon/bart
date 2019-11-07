# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.


numsrcs := $(wildcard $(srcdir)/num/*.c)
numcudasrcs := $(wildcard $(srcdir)/num/*.cu)
numobjs := $(numsrcs:.c=.o)

ifeq ($(CUDA),1)
numobjs += $(numcudasrcs:.cu=.o)
endif


.INTERMEDIATE: $(numobjs)

lib/libnum.a: libnum.a($(numobjs))


UTARGETS += test_multind test_flpmath test_splines test_linalg test_polynom test_window
UTARGETS += test_blas test_mdfft test_ops test_ops_p
UTARGETS_GPU += test_cudafft

