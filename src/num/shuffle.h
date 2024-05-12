
#include <stdlib.h>

#include "misc/cppwrap.h"

extern void md_shuffle2(int N, const long dims[__VLA(N)], const long factors[__VLA(N)],
		const long ostrs[__VLA(N)], void* out, const long istrs[__VLA(N)], const void* in, size_t size);

extern void md_shuffle(int N, const long dims[__VLA(N)], const long factors[__VLA(N)],
		void* out, const void* in, size_t size);

extern void md_decompose2(int N, const long factors[__VLA(N)],
		const long odims[__VLA(N + 1)], const long ostrs[__VLA(N + 1)], void* out,
		const long idims[__VLA(N)], const long istrs[__VLA(N)], const void* in, size_t size);

extern void md_decompose(int N, const long factors[__VLA(N)], const long odims[__VLA(N + 1)],
		void* out, const long idims[__VLA(N)], const void* in, size_t size);

extern void md_recompose2(int N, const long factors[__VLA(N)],
		const long odims[__VLA(N)], const long ostrs[__VLA(N)], void* out,
		const long idims[__VLA(N + 1)], const long istrs[__VLA(N + 1)], const void* in, size_t size);

extern void md_recompose(int N, const long factors[__VLA(N)], const long odims[__VLA(N)],
		void* out, const long idims[__VLA(N + 1)], const void* in, size_t size);

#include "misc/cppwrap.h"
