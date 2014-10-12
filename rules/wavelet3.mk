# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.




wavelet3srcs := $(wildcard $(srcdir)/wavelet3/*.c)
wavelet3objs := $(wavelet3srcs:.c=.o)

wavelet3cudasrcs := $(wildcard $(srcdir)/wavelet3/*.cu)
ifeq ($(CUDA),1)
wavelet3objs += $(wavelet3cudasrcs:.cu=.o)
endif

.INTERMEDIATE: $(wavelet3objs)

lib/libwavelet3.a: libwavelet3.a($(wavelet3objs))



