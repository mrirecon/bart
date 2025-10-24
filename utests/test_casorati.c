
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


static bool test_covariance_function(void)
{
	long kdims[4] = { 6, 6, 1, 32 };
	long dims[4] = { 24, 24, 1, 32 };

	long M = md_calc_size(4, kdims);

	complex float (*cov2)[M][M] = md_alloc(2, MD_DIMS(M, M), CFL_SIZE);
	complex float (*cov1)[M][M] = md_alloc(2, MD_DIMS(M, M), CFL_SIZE);

	complex float* data = md_alloc(4, dims, CFL_SIZE);
	md_gaussian_rand(4, dims, data);

	double time = -timestamp();
	covariance_function(kdims, M, (*cov1), dims, data);

	time += timestamp();
	debug_printf(DP_DEBUG1, "Covariance time: %es\n", time);
	time = -timestamp();

	covariance_function_fft(kdims, M, (*cov2), dims, data);

	time += timestamp();
	debug_printf(DP_DEBUG1, "FFT time: %es\n", time);

	float err = md_znrmse(2, MD_DIMS(M, M), &(*cov2)[0][0], &(*cov1)[0][0]);

	md_free(cov1);
	md_free(cov2);
	md_free(data);

	debug_printf(DP_DEBUG1, "Casorati Gram matrix error: %f\n", err);

	UT_RETURN_ASSERT(err < 1.E-6);
}

UT_REGISTER_TEST(test_covariance_function);
