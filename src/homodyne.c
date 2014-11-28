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
static float homodyne_filter(long N, float frac, long p)
{
	return (abs(2 * p - N) < 2. * (frac - 0.5) * N) ? 1. : 2.;
}

static void comp_weights(void* _data, const long pos[])
{
	struct wdata* data = _data;
	data->weights[md_calc_offset(DIMS, data->wstrs, pos) / CFL_SIZE] 
		= homodyne_filter(data->wdims[data->pfdim], data->frac, pos[data->pfdim]);
}

static void homodyne(struct wdata wdata, unsigned int flags, unsigned int N, const long dims[N],
		const long strs[N], complex float* data, const complex float* idata)
{
	long cdims[N];
	md_copy_dims(N, cdims, dims);
	// cdims[0] = cdims[1] = cdims[2] = 24;
	cdims[wdata.pfdim] = (wdata.frac - 0.5) * dims[wdata.pfdim];

	complex float* center = md_alloc(N, cdims, CFL_SIZE);
	complex float* phase = md_alloc(N, dims, CFL_SIZE);

	md_resizec(N, cdims, center, dims, idata, CFL_SIZE);
	md_resizec(N, dims, phase, cdims, center, CFL_SIZE);
	md_free(center);

	ifftuc(N, dims, flags, phase, phase);
	md_zphsr(N, dims, phase, phase);

	md_zmul2(N, dims, strs, data, strs, idata, wdata.wstrs, wdata.weights);

	ifftuc(N, dims, flags, data, data);

	md_zmulc(N, dims, data, data, phase);
	md_zreal(N, dims, data, data);

	md_free(phase);
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


	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	struct wdata wdata;
	wdata.frac = frac;
	wdata.pfdim = pfdim;
	md_select_dims(N, (1 << pfdim), wdata.wdims, dims);
	md_calc_strides(N, wdata.wstrs, wdata.wdims, CFL_SIZE);
	wdata.weights = md_alloc(N, wdata.wdims, CFL_SIZE);

	md_loop(N, wdata.wdims, &wdata, comp_weights);

	if ((1 == dims[PHS2_DIM]) || (PHS2_DIM == pfdim)) {

		homodyne(wdata, FFT_FLAGS, N, dims, strs, data, idata);

	} else {

		unsigned int pardim = PHS2_DIM;

		ifftuc(N, dims, FFT_FLAGS & ~(1 << pfdim), data, idata);

		long rdims[N];
		md_select_dims(N, ~(1 << pardim), rdims, dims);
		long rstrs[N];
		md_calc_strides(N, rstrs, rdims, CFL_SIZE);

#pragma 	omp parallel for
		for (unsigned int i = 0; i < dims[pardim]; i++) {

			complex float* tmp = md_alloc(N, rdims, CFL_SIZE);
			long pos[N];
			md_set_dims(N, pos, 0);
			pos[pardim] = i;

			md_copy_block(N, pos, rdims, tmp, dims, data, CFL_SIZE);
			homodyne(wdata, (1 << pfdim), N, rdims, rstrs, tmp, tmp);
			md_copy_block(N, pos, dims, data, rdims, tmp, CFL_SIZE);
			md_free(tmp);
		}
	}

	md_free(wdata.weights);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, data);

	exit(0);
}


