# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.



miscsrcs := $(wildcard $(srcdir)/misc/*.c)
miscobjs := $(miscsrcs:.c=.o)

.INTERMEDIATE: $(miscobjs)

lib/libmisc.a: libmisc.a($(miscobjs))




