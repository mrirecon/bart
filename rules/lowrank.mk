# Copyright 2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.



lowranksrcs := $(wildcard $(srcdir)/lowrank/*.c)
#lowranksrcs := $(wildcard $(srcdir)/lowrank/lr*.c)
lowrankobjs := $(lowranksrcs:.c=.o)

.INTERMEDIATE: $(lowrankobjs)

lib/liblowrank.a: liblowrank.a($(lowrankobjs))



