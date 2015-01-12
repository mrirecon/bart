


boxsrcs := $(BTARGETS:%=src/%.c)
boxobjs := $(boxsrcs:.c=.o)

.INTERMEDIATE: $(boxobjs)

lib/libbox.a: libbox.a($(boxobjs))
lib/libbox.a: CFLAGS += -Wno-missing-prototypes



