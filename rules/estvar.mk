# Copyright 2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.

estvarsrcs := $(wildcard $(srcdir)/estvar/*.c)
estvarobjs := $(estvarsrcs:.c=.o)

.INTERMEDIATE: $(estvarobjs)

lib/libestvar.a: libestvar.a($(estvarobjs))
