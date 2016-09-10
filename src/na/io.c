
#include <complex.h>

#include "misc/mmio.h"

#include "num/multind.h"
#include "num/iovec.h"

#include "na/na.h"

#include "io.h"

#ifndef DIMS
#define DIMS 16
#endif


static void buf_unmap(void* p, size_t size)
{
	unmap_cfl(1, MD_DIMS(size / sizeof(complex float)), p);
}


na na_load(const char* name)
{
	long dims[DIMS];
	complex float* p = load_cfl(name, DIMS, dims);

	size_t size = md_calc_size(DIMS, dims) * sizeof(complex float);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	return na_wrap2(DIMS, &dims, &strs, p, sizeof(complex float), size, buf_unmap);
}

na na_create(const char* name, ty t)
{
	assert(sizeof(complex float) == t->size);
	assert(DIMS == t->N);

	complex float* p = create_cfl(name, t->N, t->dims);

	long strs[t->N];
	md_calc_strides(t->N, strs, t->dims, t->size);

	size_t size = md_calc_size(t->N, t->dims) * t->size;

	return na_wrap_cb(t, t->N, (const long(*)[t->N])strs, p, size, buf_unmap);
}


