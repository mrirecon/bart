
#include <stdlib.h>

struct na_s;
struct iovec_s;
typedef struct na_s* na;
typedef const struct iovec_s* ty;

struct long_array_s {

	unsigned int N;
	const long* ar;
};

extern ty na_type(na x);
extern ty ty_create(unsigned int N, const long (*dims)[N], size_t size);

extern void ty_free(ty t);

extern na na_wrap(ty t, void* data);
extern na na_wrap_cb(ty t, unsigned int N, const long (*strs)[N], void* data, size_t size, void (*del)(void* data, size_t size));
extern na na_wrap2(unsigned int N, const long (*dims)[N], const long (*strs)[N], void* data, size_t elsize, size_t size, void (*del)(void* data, size_t size));

extern na na_new(unsigned int N, const long (*dims)[N], size_t size);
extern na na_inst(ty t);
extern void na_free(na x);

extern na na_view(na x);
extern na na_clone(na x);
extern na na_slice(na x, unsigned int flags, unsigned int N, const long (*pos)[N]);

extern struct long_array_s na_get_dimensions(na x);
extern struct long_array_s na_get_strides(na x);

#define LONG_ARRAY_CAST(x) ({ struct long_array_s _x = (x); (long(*)[_x.N])_x.ar; })
#define NA_DIMS(x) LONG_ARRAY_CAST(na_get_dimensions(x))
#define NA_STRS(x) LONG_ARRAY_CAST(na_get_strides(x))

extern void* na_ptr(na x);

extern size_t na_element_size(na x);

extern unsigned int na_rank(na x);
extern void na_free(na x);

extern void na_copy(na dst, na src);
extern void na_clear(na dst);


#if __GNUC__ < 5
#include "misc/pcaa.h"

#define ty_create(N, dims, size) \
	ty_create(N, AR2D_CAST(const long, 1, N, dims), size)

#define na_wrap_cb(t, N, strs, data, size, del) \
	na_wrap_cb(t, N, AR2D_CAST(const long, 1, N, strs), data, size, del)

#define na_wrap2(N, dims, strs, data, elsize, size, del) \
	na_wrap2(N, AR2D_CAST(const long, 1, N, dims), AR2D_CAST(const long, 1, N, strs), data, elsize, size, del)

#define na_slice(x, flags, N, pos) \
	na_slice(x, flags, N, AR2D_CAST(const long, 1, N, pos))

#define na_new(N, dims, size) \
	na_new(N, AR2D_CAST(const long, 1, N, dims), size)

#endif


