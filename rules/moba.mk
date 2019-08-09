

mobasrcs := $(wildcard $(srcdir)/moba/*.c)
mobaobjs := $(mobasrcs:.c=.o)

.INTERMEDIATE: $(mobaobjs)

lib/libmoba.a: libmoba.a($(mobaobjs))

UTARGETS += test_moba
MODULES_test_moba += -lmoba -lnlops -llinops



