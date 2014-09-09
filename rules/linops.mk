# Copyright 2014. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.


linopssrcs := $(wildcard $(srcdir)/linops/*.c)
linopscudasrcs := $(wildcard $(srcdir)/linops/*.cu)
linopsobjs := $(linopssrcs:.c=.o)

ifeq ($(CUDA),1)
linopsobjs += $(linopscudasrcs:.cu=.o)
endif


.INTERMEDIATE: $(linopsobjs)

lib/liblinops.a: liblinops.a($(linopsobjs))




