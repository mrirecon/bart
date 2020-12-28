/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/mdfft.h"

#include "utest.h"





static bool test_mdfft(void)
{
	enum { N = 4 };
	long dims[N] = { 10, 10, 1, 1 };

	complex float* x = md_calloc(N, dims, sizeof(complex float));
	complex float* y = md_calloc(N, dims, sizeof(complex float));

	x[0] = 1.;

	md_fft(N, dims, 3, 0, y, x);

	md_zfill(N, dims, x, 1.);

	float err = md_znrmse(N, dims, x, y);

	md_free(x);
	md_free(y);

	return (err < 1.E-6);
}




UT_REGISTER_TEST(test_mdfft);

