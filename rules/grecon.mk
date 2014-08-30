# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.




greconsrcs := $(wildcard $(srcdir)/grecon/*.c)
greconobjs := $(greconsrcs:.c=.o)

.INTERMEDIATE: $(greconobjs)

lib/libgrecon.a: libgrecon.a($(greconobjs))



