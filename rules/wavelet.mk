
waveletsrcs := $(wildcard $(srcdir)/wavelet/*.c)
waveletobjs := $(waveletsrcs:.c=.o)

waveletcudasrcs := $(wildcard $(srcdir)/wavelet/*.cu)
ifeq ($(CUDA),1)
waveletobjs += $(waveletcudasrcs:.cu=.o)
endif

.INTERMEDIATE: $(waveletobjs)

lib/libwavelet.a: libwavelet.a($(waveletobjs))


