


boxsrcs := $(XTARGETS:%=src/%.c)
boxobjs := $(boxsrcs:.c=.o)

.INTERMEDIATE: $(boxobjs)

lib/libbox.a: libbox.a($(boxobjs))



