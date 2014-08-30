# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.




ismrmsrcs := $(wildcard $(srcdir)/ismrm/*.cc)
ismrmobjs := $(ismrmsrcs:.cc=.o)

.INTERMEDIATE: $(ismrmobjs)

lib/libismrm.a: libismrm.a($(ismrmobjs))
lib/libismrm.a: CPPFLAGS += $(ISMRM_H)



