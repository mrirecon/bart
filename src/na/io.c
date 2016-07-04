
#include <complex.h>

#include "misc/mmio.h"

#include "num/multind.h"

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

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	size_t size = md_calc_size(DIMS, dims) * sizeof(complex float);

	return na_wrap_cb(DIMS, &dims, &strs, p, sizeof(complex float), size, buf_unmap);
}

na na_create(const char* name, unsigned int N, const long (*dims)[N], size_t elsize)
{
	assert(elsize == sizeof(complex float));

	complex float* p = create_cfl(name, DIMS, *dims);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, *dims, elsize);

	size_t size = md_calc_size(DIMS, *dims) * elsize;

	return na_wrap_cb(DIMS, dims, &strs, p, elsize, size, buf_unmap);
}


