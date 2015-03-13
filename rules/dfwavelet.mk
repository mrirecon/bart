# Copyright 2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.




dfwaveletsrcs := $(wildcard $(srcdir)/dfwavelet/*.c)
dfwaveletobjs := $(dfwaveletsrcs:.c=.o)

dfwaveletcudasrcs := $(wildcard $(srcdir)/dfwavelet/*.cu)
ifeq ($(CUDA),1)
dfwaveletobjs += $(dfwaveletcudasrcs:.cu=.o)
endif

.INTERMEDIATE: $(dfwaveletobjs)

lib/libdfwavelet.a: libdfwavelet.a($(dfwaveletobjs))



