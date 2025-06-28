
#include <stdbool.h>
#include <stdlib.h>

#include "misc/cppwrap.h"

struct iovec_s {
	
	int N;
	const long* dims;
	const long* strs;
	size_t size;
};


extern const struct iovec_s* iovec_create(int N, const long dims[__VLA(N)], size_t size);
extern const struct iovec_s* iovec_create2(int N, const long dims[__VLA(N)], const long strs[__VLA(N)], size_t size);
extern void iovec_free(const struct iovec_s* x);
extern bool iovec_check(const struct iovec_s* iov, int N, const long dims[__VLA(N)], const long strs[__VLA(N)]);
extern bool iovec_compare(const struct iovec_s* iova, const struct iovec_s* iovb);

// in-place initialization and deconstruction
extern void iovec_init2(struct iovec_s* n, int N, const long dims[__VLA(N)], const long strs[__VLA(N)], size_t size);
extern void iovec_destroy(const struct iovec_s* x);

extern void debug_print_iovec(int level, const struct iovec_s* vec);

#include "misc/cppwrap.h"

