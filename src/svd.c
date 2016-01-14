/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/lapack.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"



static const char usage_str[] = "<input> <U> <S> <VH>";
static const char help_str[] = "Compute singular-value-decomposition (SVD).\n";


int main_svd(int argc, char* argv[])
{
	bool econ = false;

	const struct opt_s opts[] = {

		OPT_SET('e', &econ, "econ"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);


	int N = 2;
	long dims[N];

	complex float* in = load_cfl(argv[1], N, dims);

	long dimsU[2] = { dims[0], econ ? MIN(dims[0], dims[1]) : dims[0] };
	long dimsS[2] = { MIN(dims[0], dims[1]), 1 };
	long dimsVH[2] = { econ ? MIN(dims[0], dims[1]) : dims[1], dims[1] };

	complex float* U = create_cfl(argv[2], N, dimsU);
	complex float* S = create_cfl(argv[3], N, dimsS);
	complex float* VH = create_cfl(argv[4], N, dimsVH);

	float* SF = md_alloc(2, dimsS, FL_SIZE);

	(econ ? lapack_svd_econ : lapack_svd)(dims[0], dims[1],
			MD_CAST_ARRAY2(complex float, 2, dimsU, U, 0, 1),
			MD_CAST_ARRAY2(complex float, 2, dimsVH, VH, 0, 1),
			SF, MD_CAST_ARRAY2(complex float, 2, dims, in, 0, 1));

	for (int i = 0 ; i < dimsS[0]; i++)
		S[i] = SF[i];

	md_free(SF);


	unmap_cfl(N, dims, in);
	unmap_cfl(N, dimsU, U);
	unmap_cfl(N, dimsS, S);
	unmap_cfl(N, dimsVH, VH);
	exit(0);
}


