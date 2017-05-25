/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <assert.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/iovec.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "linops/linop.h"
#include "linops/waveop.h"


#ifndef DIMS
#define DIMS 16
#endif



static const char usage_str[] = "flags [dims] <input> <output>";
static const char help_str[] = "Perform wavelet transform.";



int main_wavelet(int argc, char* argv[])
{
	bool adj = false;
        
	const struct opt_s opts[] = {

		OPT_SET('a', &adj, "adjoint (specify dims)"),
	};

	cmdline(&argc, argv, 3, 100, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	unsigned int flags = atof(argv[1]);
	unsigned int n = adj ? bitcount(flags) : 0;

	assert((int)n + 3 == argc - 1);


	const unsigned int N = DIMS;
	long idims[N];

	complex float* idata = load_cfl(argv[n + 2], N, idims);

	long dims[N];

	if (adj) {

		md_singleton_dims(N, dims);

		unsigned int j = 0;

		for (unsigned int i = 0; i < N; i++)
			if (MD_IS_SET(flags, i))
				dims[i] = atoi(argv[j++ + 2]);

		assert(j == n);
	
	} else {

		md_copy_dims(N, dims, idims);
	}

	long minsize[N];

	for (unsigned int i = 0; i < N; i++)
		minsize[i] = MD_IS_SET(flags, i) ? 16 : dims[i];

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	const struct linop_s* w = linop_wavelet3_create(N, flags, dims, strs, minsize);

	long odims[N];
	md_copy_dims(N, odims, (adj ? linop_domain : linop_codomain)(w)->dims);

	complex float* odata = create_cfl(argv[n + 3], N, odims);

	(adj ? linop_adjoint : linop_forward)(w, N, odims, odata, N, idims, idata);
	
	unmap_cfl(N, idims, idata);
	unmap_cfl(N, odims, odata);
	exit(0);
}


