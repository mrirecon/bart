
nlopssrcs := $(wildcard $(srcdir)/nlops/*.c)
nlopscudasrcs := $(wildcard $(srcdir)/nlops/*.cu)
nlopsobjs := $(nlopssrcs:.c=.o)


.INTERMEDIATE: $(nlopsobjs)

lib/libnlops.a: libnlops.a($(nlopsobjs))


UTARGETS += test_nlop
MODULES_test_nlop += -lnlops -llinops

