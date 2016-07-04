
#include <complex.h>

#include "misc/mmio.h"

#include "num/multind.h"

#include "na/na.h"

#include "io.h"

#ifndef DIMS
#define DIMS 16
#endif


na na_load(const char* name)
{
	long dims[DIMS];
	complex float* p = load_cfl(name, DIMS, dims);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, sizeof(complex float));

	return na_wrap(DIMS, &dims, &strs, p, sizeof(complex float));
}

na na_create(const char* name, unsigned int N, const long (*dims)[N], size_t size)
{
	assert(size == sizeof(complex float));

	complex float* p = create_cfl(name, DIMS, *dims);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, *dims, sizeof(complex float));

	return na_wrap(DIMS, dims, &strs, p, sizeof(complex float));
}


