
#include <complex.h>
#include <stdint.h>

#include "misc/debug.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "calib/calmat.h"

#include "linops/linop.h"
#include "linops/casorati.h"
#include "linops/lintest.h"

#include "num/rand.h"

#include "utest.h"


static bool test_linop_casorati(void)
{
	enum { N = 4 };

	const long dims[N] = { 12, 1, 12, 8 };
	const long kdim[N] = {  5, 5, 12, 1 };

	complex float* data = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, data);

	const struct linop_s* op = linop_casorati_create(N, kdim, dims, data);
	md_free(data);

	float err_normal = linop_test_normal(op);
	float err_adjoint = linop_test_adjoint(op);
	linop_free(op);

	UT_RETURN_ASSERT(err_normal < UT_TOL && err_adjoint < UT_TOL);
}

UT_REGISTER_TEST(test_linop_casorati);

static bool test_linop_casoratiH(void)
{
	enum { N = 4 };

	const long dims[N] = { 12, 1, 12, 8 };
	const long kdim[N] = {  5, 5, 12, 1 };

	complex float* data = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, data);

	const struct linop_s* op = linop_casoratiH_create(N, kdim, dims, data);
	md_free(data);

	float err_normal = linop_test_normal(op);
	float err_adjoint = linop_test_adjoint(op);
	linop_free(op);

	UT_RETURN_ASSERT(err_normal < UT_TOL && err_adjoint < UT_TOL);
}

UT_REGISTER_TEST(test_linop_casoratiH);


