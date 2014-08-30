# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.



noirsrcs := $(wildcard $(srcdir)/noir/*.c)
noirobjs := $(noirsrcs:.c=.o)

.INTERMEDIATE: $(noirobjs)

lib/libnoir.a: libnoir.a($(noirobjs))




