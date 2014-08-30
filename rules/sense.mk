# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.




sensesrcs := $(wildcard $(srcdir)/sense/*.c)
senseobjs := $(sensesrcs:.c=.o)

.INTERMEDIATE: $(senseobjs)

lib/libsense.a: libsense.a($(senseobjs))



