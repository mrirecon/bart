/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/casorati.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE	sizeof(complex float)
#endif

static const char help_str[] = "Casorati matrix with kernel (kern1, ..., kernN) along dimensions (dim1, ..., dimN).";


int main_casorati(int argc, char* argv[argc])
{
	long count = 0;
	unsigned int* dims = NULL;
	unsigned int* kerns = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 2, { OPT_UINT, sizeof(*dims), &dims, "dim" },
					   { OPT_UINT, sizeof(*kerns), &kerns, "kern" }),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long idims[DIMS];
	long kdims[DIMS];
	long odims[2];

	complex float* idata = load_cfl(in_file, DIMS, idims);

	md_copy_dims(DIMS, kdims, idims);

	for (int i = 0; i < count; i++) {

		unsigned int kdim = dims[i];
		unsigned int ksize = kerns[i];

		assert(kdim < DIMS);
		assert(ksize >= 1);

		kdims[kdim] = ksize;
	}


	casorati_dims(DIMS, odims, kdims, idims);

	complex float* odata = create_cfl(out_file, 2, odims);

	long istrs[DIMS];
	md_calc_strides(DIMS, istrs, idims, CFL_SIZE);

	casorati_matrix(DIMS, kdims, odims, odata, idims, istrs, idata);

	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(2, odims, odata);

	xfree(dims);
	xfree(kerns);

	return 0;
}
