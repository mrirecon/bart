# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.




wavelet2srcs := $(wildcard $(srcdir)/wavelet2/*.c)
wavelet2objs := $(wavelet2srcs:.c=.o)

wavelet2cudasrcs := $(wildcard $(srcdir)/wavelet2/*.cu)
ifeq ($(CUDA),1)
wavelet2objs += $(wavelet2cudasrcs:.cu=.o)
endif

.INTERMEDIATE: $(wavelet2objs)

lib/libwavelet2.a: libwavelet2.a($(wavelet2objs))



