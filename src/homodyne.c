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
#include "num/init.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"





static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-I] [-C] [-P phase_ref] [-r alpha] dim fraction <input> <output>\n", name);
}

static void help(const char* name, FILE *fd)
{
	usage(name, fd);
	fprintf(fd, "\nPerform homodyne reconstruction along dimension dim.\n"
		"\t-C\tClear unacquired portion of kspace\n"
		"\t-I\tInput is in image domain\n"
		"\t-P <phase_ref>\tUse <phase_ref> as phase reference\n"
		"\t-r <alpha>\tOffset of ramp filter, between 0 and 1\n"
		"\t-h\thelp\n");
}



struct wdata {

	float frac;
	float alpha;
	int pfdim;
	long wdims[DIMS];
	long wstrs[DIMS];
	complex float* weights;

	bool clear;
};


/**
 * Applies the Homodyne filter.
 * @param N k-space dimension
 * @param p k-space position
 * @param frac is the fraction of acquired k-space
 * @param alpha is the offset of the ramp, between 0 and 1
 * @param clear clear acquired k-space
 *
 * The ramp portion is given by 2*(alpha - 1) / (end - start) * (p - end) + alpha
 * alpha = 0 is a full ramp, alpha = 1 is a horizontal line
 */
static float homodyne_filter(long N, float frac, float alpha, bool clear, long p)
{
	if (frac <= 0.5)
		return 1.;

	float start = N * (1 - frac);
	float end = N * frac;

	float ret = clear ? 0. : 1.;


	if (p < start)
		ret = 2.;
	else if (p >= start && p < end)
		ret = 2 * (alpha - 1) / (end - start) * (p - end) + alpha;

	return ret;
}

static void comp_weights(void* _data, const long pos[])
{
	struct wdata* data = _data;
	data->weights[md_calc_offset(DIMS, data->wstrs, pos) / CFL_SIZE] 
		= homodyne_filter(data->wdims[data->pfdim], data->frac, data->alpha, data->clear, pos[data->pfdim]);
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
	bool image = false;
	const char* phase_ref = NULL;

	float alpha = 0.;

	num_init();

	int com;
	while (-1 != (com = getopt(argc, argv, "r:ICP:h"))) {

		switch (com) {

		case 'r':
			alpha = atof(optarg);
			break;

		case 'I':
			image = true;
			break;

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

	if (image) {
		complex float* ksp_in = md_alloc(N, dims, CFL_SIZE);
		fftuc(N, dims, FFT_FLAGS, ksp_in, idata);
		md_copy(N, dims, idata, ksp_in, CFL_SIZE);
		md_free(ksp_in);
	}


	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	struct wdata wdata;
	wdata.frac = frac;
	wdata.pfdim = pfdim;
	md_select_dims(N, MD_BIT(pfdim), wdata.wdims, dims);
	md_calc_strides(N, wdata.wstrs, wdata.wdims, CFL_SIZE);
	wdata.weights = md_alloc(N, wdata.wdims, CFL_SIZE);
	wdata.alpha = alpha;
	wdata.clear = clear;

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

	homodyne(wdata, FFT_FLAGS, N, dims, strs, data, idata, pstrs, phase);

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


