
#include <stdlib.h>

struct na_s;
typedef struct na_s* na;

struct long_array_s {

	unsigned int N;
	const long* ar;
};


extern na na_wrap(unsigned int N, const long (*dims)[N], const long (*strs)[N], void* data, size_t size);
extern na na_new(unsigned int N, const long (*dims)[N], size_t size);
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


