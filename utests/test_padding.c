
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/lintest.h"
#include "linops/someops.h"

#include "utest.h"



static bool test_padding(void)
{
	enum { N = 2 };
	long dims_in[N] = { 3, 2 };
	long dims_out[N] = { 7, 4 };

	long pad[] = { 2, 1 };

	complex float in[] = {
		1, 2, 3,
		4, 5, 6,
	};

	complex float exp_valid[] = {
		1, 2, 3,
		4, 5, 6
	};

	complex float exp_same[] = {
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 2, 3, 0, 0,
		0, 0, 4, 5, 6, 0, 0,
		0, 0, 0, 0, 0, 0, 0
	};

	complex float exp_reflect[] = {
		6, 5, 4, 5, 6, 5, 4,
		3, 2, 1, 2, 3, 2, 1,
		6, 5, 4, 5, 6, 5, 4,
		3, 2, 1, 2, 3, 2, 1
	};

	complex float exp_sym[] = {
		2, 1, 1, 2, 3, 3, 2,
		2, 1, 1, 2, 3, 3, 2,
		5, 4, 4, 5, 6, 6, 5,
		5, 4, 4, 5, 6, 6, 5
	};

	complex float exp_cyc[] = {
		5, 6, 4, 5, 6, 4, 5,
		2, 3, 1, 2, 3, 1, 2,
		5, 6, 4, 5, 6, 4, 5,
		2, 3, 1, 2, 3, 1, 2
	};

	complex float* out = md_alloc(2, dims_out, CFL_SIZE);

	const struct linop_s* lin_pad;
	float err = 0;

	lin_pad = linop_padding_create(2, dims_in, PAD_SAME, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_same, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_REFLECT, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_reflect, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_SYMMETRIC, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_sym, out);

	lin_pad = linop_padding_create(2, dims_in, PAD_CYCLIC, pad, pad);
	linop_forward_unchecked(lin_pad, out, in);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_out, exp_cyc, out);

	long pad_down[] = { -2, -1 };

	lin_pad = linop_padding_create(2, dims_out, PAD_VALID, pad_down, pad_down);
	linop_forward_unchecked(lin_pad, in, out);
	linop_free(lin_pad);
	err += md_zrmse(2, dims_in, in, exp_valid);

	md_free(out);

	UT_ASSERT(1.e-7 > err);
}

UT_REGISTER_TEST(test_padding);


static bool test_padding_adjoint(void)
{
	enum { N = 2 };
	long dims_in[N] = { 3, 2 };
	long dims_out[N] = { 7, 4 };
	long pad[] = { 2, 1 };

	const struct linop_s* lin_pad;
	float err = 0;

	lin_pad = linop_padding_create(2, dims_in, PAD_SAME, pad, pad);
	err += linop_test_adjoint(lin_pad);
	linop_free(lin_pad);

	lin_pad = linop_padding_create(2, dims_in, PAD_REFLECT, pad, pad);
	linop_free(lin_pad);

	lin_pad = linop_padding_create(2, dims_in, PAD_SYMMETRIC, pad, pad);
	err += linop_test_adjoint(lin_pad);
	linop_free(lin_pad);

	lin_pad = linop_padding_create(2, dims_in, PAD_CYCLIC, pad, pad);
	err += linop_test_adjoint(lin_pad);
	linop_free(lin_pad);

	long pad_down[] = { -2, -1 };

	lin_pad = linop_padding_create(2, dims_out, PAD_VALID, pad_down, pad_down);
	err += linop_test_adjoint(lin_pad);
	linop_free(lin_pad);

	debug_printf(DP_DEBUG1, "err: %.8f\n", err);

	UT_ASSERT(2.e-6 > err);
}

UT_REGISTER_TEST(test_padding_adjoint);



