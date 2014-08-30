# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.



sakesrcs := $(wildcard $(srcdir)/sake/*.c)
sakeobjs := $(sakesrcs:.c=.o)

.INTERMEDIATE: $(sakeobjs)

lib/libsake.a: libsake.a($(sakeobjs))



