# Copyright 2016. Martin Uecker.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.


nasrcs := $(wildcard $(srcdir)/na/*.c)
naobjs := $(nasrcs:.c=.o)

.INTERMEDIATE: $(naobjs)

lib/libna.a: libna.a($(naobjs))


