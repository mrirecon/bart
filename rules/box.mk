


boxsrcs := $(BTARGETS:%=src/%.c)
boxobjs := $(boxsrcs:.c=.o)

.INTERMEDIATE: $(boxobjs)

lib/libbox.a: libbox.a($(boxobjs))


box2srcs := $(XTARGETS:%=src/%.c)
box2objs := $(box2srcs:.c=.o)

.INTERMEDIATE: $(box2objs)

lib/libbox2.a: libbox2.a($(box2objs))




