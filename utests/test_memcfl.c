
#include <complex.h>

#include "utest.h"

#include "misc/mmio.h"
#include "misc/memcfl.h"


static bool test_memcfl_load(void)
{
	long dims[2] = { 10, 5 };
	complex float* x = memcfl_create("test.mem", 2, dims);

	for (int i = 0; i < 50; i++)
		x[i] = i;

	unmap_cfl(2, dims, x);

	long dims2[2];
	complex float* y = load_cfl("test.mem", 2, dims2);

	if (!((dims[0] == dims2[0]) && dims[1] == dims2[1]))
		return false;

	for (int i = 0; i < 50; i++)
		if (x[i] != i)
			return false;

	unmap_cfl(2, dims, y);

	memcfl_unlink("test.mem");

	return true;
}


UT_REGISTER_TEST(test_memcfl_load);


static bool test_memcfl_write(void)
{
	long dims[2] = { 10, 5 };
	complex float* x = create_cfl("test.mem", 2, dims);

	for (int i = 0; i < 50; i++)
		x[i] = i;

	unmap_cfl(2, dims, x);

	long dims2[2];
	complex float* y = memcfl_load("test.mem", 2, dims2);

	if (!((dims[0] == dims2[0]) && dims[1] == dims2[1]))
		return false;

	for (int i = 0; i < 50; i++)
		if (x[i] != i)
			return false;

	unmap_cfl(2, dims, y);

	memcfl_unlink("test.mem");

	return true;
}


UT_REGISTER_TEST(test_memcfl_write);


