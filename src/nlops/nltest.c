/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "misc/debug.h"

#include "num/rand.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#if 1
#include "nlops/nlop.h"

#include "nltest.h"


float nlop_test_derivative(const struct nlop_s* op)
{
	int N_dom = nlop_domain(op)->N;
	int N_cod = nlop_codomain(op)->N;

	long dims_dom[N_dom];
	md_copy_dims(N_dom, dims_dom, nlop_domain(op)->dims);

	long dims_cod[N_cod];
	md_copy_dims(N_cod, dims_cod, nlop_codomain(op)->dims);

	complex float* h = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* x1 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* x2 = md_alloc(N_dom, dims_dom, CFL_SIZE);
	complex float* d1 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* d2 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* d3 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* y1 = md_alloc(N_cod, dims_cod, CFL_SIZE);
	complex float* y2 = md_alloc(N_cod, dims_cod, CFL_SIZE);

	md_gaussian_rand(N_dom, dims_dom, x1);
	md_gaussian_rand(N_dom, dims_dom, h);


	nlop_apply(op, N_cod, dims_cod, y1, N_dom, dims_dom, x1);
	nlop_derivative(op, N_cod, dims_cod, d1, N_dom, dims_dom, h);

	float scale = 1.;
	float val0 = 0.;
	float val = 0.;
	float vall = 0.;

	for (int i = 0; i < 10; i++) {

		// d = F(x + s * h) - F(x)
		md_copy(N_dom, dims_dom, x2, x1, CFL_SIZE);
		md_zaxpy(N_dom, dims_dom, x2, scale, h);
		nlop_apply(op, N_cod, dims_cod, y2, N_dom, dims_dom, x2);
		md_zsub(N_cod, dims_cod, d2, y2, y1);

		// DF(s * h)
		md_zsmul(N_cod, dims_cod, d3, d1, scale);
		md_zsub(N_cod, dims_cod, d2, d2, d3);

		val = md_znorm(N_cod, dims_cod, d2);

		debug_printf(DP_DEBUG1, "%f/%f=%f\n", val, scale, val / scale);

		val /= scale;

		if ((0 == i) || (val > vall))
			val0 = val;

		vall = val;
		scale /= 2.;
	}


	md_free(h);
	md_free(x1);
	md_free(x2);
	md_free(y1);
	md_free(y2);
	md_free(d1);
	md_free(d2);
	md_free(d3);

	return val / val0;
}


#endif
