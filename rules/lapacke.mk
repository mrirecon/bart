
lapackesrcs := $(wildcard $(srcdir)/lapacke/*.c)
lapackeobjs := $(lapackesrcs:.c=.o)

.INTERMEDIATE: $(lapackeobjs)

lib/liblapacke.a: liblapacke.a($(lapackeobjs))


