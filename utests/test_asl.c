
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "iter/asl.h"

#include "utest.h"


static bool test_asl(void)
{
	enum { N = 16 };
	long dims[N] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 128;
	dims[1] = 128;	

	int asl_dim = ITER_DIM;
	
	long asl_dims[DIMS];
	md_copy_dims(DIMS, asl_dims, dims);
	asl_dims[asl_dim] = 2;

	complex float* src1 = md_alloc(DIMS, dims, CFL_SIZE);
	complex float* src2 = md_alloc(DIMS, dims, CFL_SIZE);
	
	md_zfill(DIMS, dims, src1, 1.);
	md_zfill(DIMS, dims, src2, 0.7);

	complex float* src = md_alloc(DIMS, asl_dims, CFL_SIZE);
	md_zfill(DIMS, asl_dims, src, 0);

	long pos[DIMS] = { 0 };
	md_copy_block(DIMS, pos, asl_dims, src, dims, src1, CFL_SIZE);
	pos[asl_dim] = 1;
	md_copy_block(DIMS, pos, asl_dims, src, dims, src2, CFL_SIZE);

	complex float* dst = md_alloc(DIMS, dims, CFL_SIZE);

	complex float* ref = md_alloc(DIMS, dims, CFL_SIZE);
	md_zfill(DIMS, dims, ref, 0.3);

	const struct linop_s* op = linop_asl_create(DIMS, asl_dims, asl_dim);
	linop_forward(op, DIMS, dims, dst, DIMS, asl_dims, src);
	linop_free(op);

	float err = md_znrmse(DIMS, dims, dst, ref);

	md_free(src);
	md_free(dst);
	md_free(ref);

	md_free(src1);
	md_free(src2);

	UT_RETURN_ASSERT(UT_TOL > err);
}

UT_REGISTER_TEST(test_asl);


static bool test_hadamard_encoding(void)
{
	enum { N = 4 };
	long idims[N] = { 2, 2, 4, 1 };

	int had_dim = 2;
	
	complex float* src = md_alloc(N, idims, CFL_SIZE);
	complex float* dst = md_alloc(N, idims, CFL_SIZE);

	md_zfill(N, idims, src, 1.);
	
	complex float* ref = md_alloc(N, idims, CFL_SIZE);
	md_zfill(N, idims, ref, 0.0f + 0.0f * I);

	ref[0] = -2.0f + 0.0f * I;
	ref[1] = -2.0f + 0.0f * I;
	ref[2] = -2.0f + 0.0f * I;
	ref[3] = -2.0f + 0.0f * I;

	struct linop_s* op = linop_hadamard_create(N, idims, had_dim);
	linop_forward(op, N, idims, dst, N, idims, src);
	linop_free(op);

	float err = md_znrmse(N, idims, dst, ref);

	md_free(src);
	md_free(dst);
	md_free(ref);

	UT_RETURN_ASSERT(UT_TOL > err);
}

UT_REGISTER_TEST(test_hadamard_encoding);
