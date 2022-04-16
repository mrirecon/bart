/* Copyright 2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/conv.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"


static const char help_str[] = "Conway's game of life.";



int main_conway(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	int iter = 20;
	bool periodic = false;

	const struct opt_s opts[] = {

		OPT_SET('P', &periodic, "periodic boundary conditions"),
		OPT_INT('n', &iter, "#", "nr. of iterations"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[2];

	complex float* init = load_cfl(in_file, 2, dims);

	complex float* world = md_alloc(2, dims, CFL_SIZE);

	md_copy(2, dims, world, init, CFL_SIZE);

	unmap_cfl(2, dims, init);

	long wdims[3];
	md_copy_dims(2, wdims, dims);
	wdims[2] = 1;

	long odims[3];
	md_copy_dims(2, odims, dims);
	odims[2] = iter;

	complex float* out = create_cfl(out_file, 3, odims);

	long mdims[2] = { 3, 3 };

	complex float mask[3][3] = {
		{ 1., 1., 1., },
		{ 1., 0., 1., },
		{ 1., 1., 1., },
	};

	complex float* buf = md_alloc(2, dims, CFL_SIZE);
	complex float* tmp = md_alloc(2, dims, CFL_SIZE);

	struct conv_plan* plan = conv_plan(2, 3UL, periodic ? CONV_CYCLIC : CONV_TRUNCATED, CONV_SYMMETRIC, dims, dims, mdims, &mask[0][0]);

	for (int i = 0; i < iter; i++) {

		conv_exec(plan, buf, world);

		md_zslessequal(2, dims, tmp, buf, 3.1);
		md_zadd(2, dims, buf, buf, world);
		md_zsgreatequal(2, dims, world, buf, 2.9);
		md_zmul(2, dims, world, world, tmp);

		md_copy_block(3, (long[3]){ [2] = i }, odims, out, wdims, world, CFL_SIZE);
	}

	conv_free(plan);

	md_free(buf);
	md_free(tmp);

	unmap_cfl(3, odims, out);

	return 0;
}


