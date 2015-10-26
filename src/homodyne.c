/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <unistd.h>


#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"





static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-C] [-P phase_ref] dim fraction <input> <output>\n", name);
}

static void help(const char* name, FILE *fd)
{
	usage(name, fd);
	fprintf(fd, "\nPerform homodyne reconstruction along dimension dim.\n"
		"\t-C\tClear unacquired portion of kspace\n"
		"\t-P <phase_ref>\tUse <phase_ref> as phase reference\n"
		"\t-h\thelp\n");
}



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

static complex float* estimate_phase(struct wdata wdata, unsigned int flags,
		unsigned int N, const long dims[N], const complex float* idata)
{

	long cdims[N];
	md_copy_dims(N, cdims, dims);
	// cdims[0] = cdims[1] = cdims[2] = 24;
	cdims[wdata.pfdim] = (wdata.frac - 0.5) * dims[wdata.pfdim];

	complex float* center = md_alloc(N, cdims, CFL_SIZE);
	complex float* phase = md_alloc(N, dims, CFL_SIZE);

	md_resize_center(N, cdims, center, dims, idata, CFL_SIZE);
	md_resize_center(N, dims, phase, cdims, center, CFL_SIZE);
	md_free(center);

	ifftuc(N, dims, flags, phase, phase);
	md_zphsr(N, dims, phase, phase);

	return phase;
}

static void homodyne(struct wdata wdata, unsigned int flags, unsigned int N, const long dims[N],
		const long strs[N], complex float* data, const complex float* idata,
		const long pstrs[N], const complex float* phase)
{
	md_zmul2(N, dims, strs, data, strs, idata, wdata.wstrs, wdata.weights);
	ifftuc(N, dims, flags, data, data);

	md_zmulc2(N, dims, strs, data, strs, data, pstrs, phase);
	md_zreal(N, dims, data, data);
}



int main_homodyne(int argc, char* argv[])
{
	bool clear = false;
	const char* phase_ref = NULL;

	int com;
	while (-1 != (com = getopt(argc, argv, "hCP:"))) {

		switch (com) {

		case 'C':
			clear = true;
			break;

		case 'P':
			phase_ref = strdup(optarg);
			break;

		case 'h':
			help(argv[0], stdout);
			exit(0);

		default:
			help(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 4) {
		usage(argv[0], stderr);
		exit(1);
	}

	const int N = DIMS;
	long dims[N];
	complex float* idata = load_cfl(argv[optind + 2], N, dims);
	complex float* data = create_cfl(argv[optind + 3], N, dims);

	int pfdim = atoi(argv[optind + 0]);
	float frac = atof(argv[optind + 1]);

	assert((0 <= pfdim) && (pfdim < N));
	assert(frac > 0.);


	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	struct wdata wdata;
	wdata.frac = frac;
	wdata.pfdim = pfdim;
	md_select_dims(N, MD_BIT(pfdim), wdata.wdims, dims);
	md_calc_strides(N, wdata.wstrs, wdata.wdims, CFL_SIZE);
	wdata.weights = md_alloc(N, wdata.wdims, CFL_SIZE);

	md_loop(N, wdata.wdims, &wdata, comp_weights);

	long pstrs[N];
	long pdims[N];
	complex float* phase = NULL;

	if (NULL == phase_ref) {

		phase = estimate_phase(wdata, FFT_FLAGS, N, dims, idata);
		md_copy_dims(N, pdims, dims);
	}
	else
		phase = load_cfl(phase_ref, N, pdims);

	md_calc_strides(N, pstrs, pdims, CFL_SIZE);

	complex float* cdata = NULL;
	complex float* idata2 = NULL;

	if (clear) {

		long cdims[N];
		md_select_dims(N, ~MD_BIT(pfdim), cdims, dims);
		cdims[pfdim] = (int)(dims[pfdim] * frac);

		cdata = md_alloc(N, cdims, CFL_SIZE);
		idata2 = anon_cfl(NULL, N, dims);

		md_resize(N, cdims, cdata, dims, idata, CFL_SIZE);
		md_resize(N, dims, idata2, cdims, cdata, CFL_SIZE);

		md_free(cdata);
		unmap_cfl(N, dims, idata);
		idata = idata2;

	}


	if ((1 == dims[PHS2_DIM]) || (PHS2_DIM == pfdim)) {

		homodyne(wdata, FFT_FLAGS, N, dims, strs, data, idata, pstrs, phase);

	} else {

		unsigned int pardim = PHS2_DIM;

		ifftuc(N, dims, MD_CLEAR(FFT_FLAGS, pfdim), data, idata);

		long rdims[N];
		md_select_dims(N, ~MD_BIT(pardim), rdims, dims);
		long rstrs[N];
		md_calc_strides(N, rstrs, rdims, CFL_SIZE);

#pragma 	omp parallel for
		for (unsigned int i = 0; i < dims[pardim]; i++) {

			complex float* tmp = md_alloc(N, rdims, CFL_SIZE);
			long pos[N];
			md_set_dims(N, pos, 0);
			pos[pardim] = i;

			md_copy_block(N, pos, rdims, tmp, dims, data, CFL_SIZE);
			homodyne(wdata, MD_BIT(pfdim), N, rdims, rstrs, tmp, tmp, pstrs, phase);
			md_copy_block(N, pos, dims, data, rdims, tmp, CFL_SIZE);
			md_free(tmp);
		}
	}

	md_free(wdata.weights);
	if (NULL == phase_ref)
		md_free(phase);
	else {
		unmap_cfl(N, pdims, phase);
		free((void*)phase_ref);
	}

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, data);

	exit(0);
}


