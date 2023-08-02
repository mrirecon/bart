#include <math.h>
#include <complex.h>

#include "misc/misc.h"
#include "num/rand.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "utest.h"


static bool test_var(void)
{
	enum { N = 1};
	const long dims[N] = { 1000 };

	complex float* data = md_alloc(N, dims, CFL_SIZE);
	complex float var;

	md_gaussian_rand(N, dims, data);	
	md_zvar(N, dims, ~0, &var, data);
	
	md_free(data);

	UT_ASSERT(cabsf(var - 2) < 0.2);
}

UT_REGISTER_TEST(test_var);


