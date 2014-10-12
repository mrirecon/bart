/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <complex.h>


#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"





static const char* usage_str = "dim fraction <input> <output>";
static const char* help_str = "Perform homodyne reconstruction along dimension dim.\n";



struct wdata {

	float frac;
	int pfdim;
	long wdims[DIMS];
	long wstrs[DIMS];
	complex float* weights;
};


// FIXME: should we clear the side we do not use? 
static float homodyne(long N, float frac, long p)
{
	return (abs(2 * p - N) < 2. * (frac - 0.5) * N) ? 1. : 2.;
}

static void comp_weights(void* _data, const long pos[])
{
	struct wdata* data = _data;
	data->weights[md_calc_offset(DIMS, data->wstrs, pos) / CFL_SIZE] 
		= homodyne(data->wdims[data->pfdim], data->frac, pos[data->pfdim]);
}




int main_homodyne(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	const int N = DIMS;
	long dims[N];
	complex float* idata = load_cfl(argv[3], N, dims);
	complex float* data = create_cfl(argv[4], N, dims);

	int pfdim = atoi(argv[1]);
	float frac = atof(argv[2]);

	assert((0 <= pfdim) && (pfdim < N));
	assert(frac > 0.);

	unsigned int flags = 7;

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	struct wdata wdata;
	wdata.frac = frac;
	wdata.pfdim = pfdim;
	md_select_dims(N, ~0, wdata.wdims, dims); // should e only pdfim
	md_calc_strides(N, wdata.wstrs, wdata.wdims, CFL_SIZE);
	wdata.weights = md_alloc(N, wdata.wdims, CFL_SIZE);

	md_loop(N, wdata.wdims, &wdata, comp_weights);

	long cdims[N];
	md_copy_dims(N, cdims, dims);
	cdims[pfdim] = (frac - 0.5) * dims[pfdim];

	complex float* center = md_alloc(N, cdims, CFL_SIZE);
	complex float* phase = md_alloc(N, dims, CFL_SIZE);

	md_resizec(N, cdims, center, dims, idata, CFL_SIZE);
	md_resizec(N, dims, phase, cdims, center, CFL_SIZE);
	free(center);

	ifftuc(N, dims, flags, phase, phase);

	md_zphsr(N, dims, phase, phase);

	md_zmul2(N, dims, strs, data, strs, idata, wdata.wstrs, wdata.weights);

	ifftuc(N, dims, flags, data, data);

	md_zmulc(N, dims, data, data, phase);
	md_zreal(N, dims, data, data);


	free(phase);
	free(wdata.weights);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, data);

	exit(0);
}


