# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.



calibsrcs := $(wildcard $(srcdir)/calib/*.c)
calibobjs := $(calibsrcs:.c=.o)
ifeq ($(CUDA),1)
calibcudasrcs += $(wildcard $(srcdir)/calib/*.cu)
calibobjs += $(calibcudasrcs:.cu=.o)
endif


.INTERMEDIATE: $(calibobjs)

lib/libcalib.a: libcalib.a($(calibobjs))




