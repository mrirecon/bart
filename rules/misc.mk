# Copyright 2013. The Regents of the University of California.
# Copyright 2015. Martin Uecker.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.



miscsrcs := $(wildcard $(srcdir)/misc/*.c)
miscobjs := $(miscsrcs:.c=.o)

.INTERMEDIATE: $(miscobjs)

lib/libmisc.a: libmisc.a($(miscobjs))


version.new:
	@echo 'VERSION($(shell $(root)/git-version.sh))' > $@

.PHONY: version.new

$(srcdir)/misc/version.inc: version.new
	cmp -s version.new $(srcdir)/misc/version.inc || mv version.new $(srcdir)/misc/version.inc
	rm -f version.new


$(srcdir)/misc/version.o: $(srcdir)/misc/version.inc


