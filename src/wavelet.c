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



static const char help_str[] = "Perform wavelet transform.";



int main_wavelet(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	long count = 0;
	long* adims = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "bitmask"),
		ARG_TUPLE(false, &count, 1, TUPLE_LONG(&adims, "dim")),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool adj = false;
	enum wtype wtype = WAVELET_DAU2;
        
	const struct opt_s opts[] = {

		OPT_SET('a', &adj, "adjoint (specify dims)"),
		OPT_SELECT('H', enum wtype, &wtype, WAVELET_HAAR, "type: Haar"),
		OPT_SELECT('D', enum wtype, &wtype, WAVELET_DAU2, "type: Dau2"),
		OPT_SELECT('C', enum wtype, &wtype, WAVELET_CDF44, "type: CDF44"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (NULL != adims)
		adj = true;

	num_init();

	int n = adj ? bitcount(flags) : 0;

	assert(n == count);


	const int N = DIMS;
	long idims[N];

	complex float* idata = load_cfl(in_file, N, idims);

	long dims[N];
	md_copy_dims(N, dims, idims);

	if (adj) {

		int j = 0;

		for (int i = 0; i < N; i++)
			if (MD_IS_SET(flags, i))
				dims[i] = adims[j++];

		assert(j == n);
	}

	long minsize[N];

	for (int i = 0; i < N; i++)
		minsize[i] = MD_IS_SET(flags, i) ? 16 : dims[i];

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	const struct linop_s* w = linop_wavelet_create(N, flags, dims, strs, wtype, minsize, false);

	long odims[N];
	md_copy_dims(N, odims, (adj ? linop_domain : linop_codomain)(w)->dims);

	complex float* odata = create_cfl(out_file, N, odims);

	(adj ? linop_adjoint : linop_forward)(w, N, odims, odata, N, idims, idata);
	
	linop_free(w);

	unmap_cfl(N, idims, idata);
	unmap_cfl(N, odims, odata);

	xfree(adims);

	return 0;
}


