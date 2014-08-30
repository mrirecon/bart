# Copyright 2014. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.



noncartsrcs := $(wildcard $(srcdir)/noncart/*.c)
noncartobjs := $(noncartsrcs:.c=.o)

.INTERMEDIATE: $(noncartobjs)

lib/libnoncart.a: libnoncart.a($(noncartobjs))



