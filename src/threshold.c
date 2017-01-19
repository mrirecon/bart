/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2013-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015-2016 Jon Tamir <jtamir@eecs.berkeley.edu>
 * 2015 Frank Ong <frankong@berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/init.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "lowrank/lrthresh.h"

#include "linops/waveop.h"

#include "dfwavelet/prox_dfwavelet.h"

// FIXME: lowrank interface should not be coupled to mri.h -- it should take D as an input
#ifndef DIMS
#define DIMS 16
#endif



// FIXME: consider moving this to a more accessible location?
static void wthresh(unsigned int D, const long dims[D], float lambda, unsigned int flags, complex float* out, const complex float* in)
{
	long minsize[D];
	md_singleton_dims(D, minsize);

	long course_scale[3] = MD_INIT_ARRAY(3, 16);
	md_min_dims(3, ~0u, minsize, dims, course_scale);

	long strs[D];
	md_calc_strides(D, strs, dims, CFL_SIZE);

	const struct linop_s* w = linop_wavelet3_create(D, 7, dims, strs, minsize);
	const struct operator_p_s* p = prox_unithresh_create(D, w, lambda, flags, false);

	operator_p_apply(p, 1., D, dims, out, D, dims, in);

	operator_p_free(p);
}


static void lrthresh(unsigned int D, const long dims[D], int llrblk, float lambda, unsigned int flags, complex float* out, const complex float* in)
{
	long blkdims[MAX_LEV][D];

	int levels = llr_blkdims(blkdims, ~flags, dims, llrblk);
	UNUSED(levels);

	const struct operator_p_s* p = lrthresh_create(dims, false, ~flags, (const long (*)[])blkdims, lambda, false, false, false);

	operator_p_apply(p, 1., D, dims, out, D, dims, in);

	operator_p_free(p);
}


static void dfthresh(unsigned int D, const long dims[D], float lambda, complex float* out, const complex float* in)
{
	long minsize[3];
	md_singleton_dims(3, minsize);

	long coarse_scale[3] = MD_INIT_ARRAY(3, 16);
	md_min_dims(3, ~0u, minsize, dims, coarse_scale);

        complex float res[3];
        res[0] = 1.;
        res[1] = 1.;
        res[2] = 1.;

        assert(3 == dims[TE_DIM]);

        const struct operator_p_s* p = prox_dfwavelet_create(dims, minsize, res, TE_DIM, lambda, false);

	operator_p_apply(p, 1., D, dims, out, D, dims, in);

	operator_p_free(p);
}

static void hard_thresh(unsigned int D, const long dims[D], float lambda, complex float* out, const complex float* in)
{
	long size = md_calc_size(DIMS, dims) * 2;

	const float* inf = (const float*)in;
	float* outf = (float*)out;

#pragma omp parallel for
	for (long i = 0; i < size; i++)
		outf[i] = inf[i] > lambda ? inf[i] : 0.;
}



static const char usage_str[] = "lambda <input> <output>";
static const char help_str[] = "Perform (soft) thresholding with parameter lambda.";



int main_threshold(int argc, char* argv[])
{
	unsigned int flags = 0;
        
	enum th_type { NONE, WAV, LLR, DFW, MPDFW, HARD } th_type = NONE;
	int llrblk = 8;


	const struct opt_s opts[] = {

		OPT_SELECT('H', enum th_type, &th_type, HARD, "hard thresholding"),
		OPT_SELECT('W', enum th_type, &th_type, WAV, "daubechies wavelet soft-thresholding"),
		OPT_SELECT('L', enum th_type, &th_type, LLR, "locally low rank soft-thresholding"),
		OPT_SELECT('D', enum th_type, &th_type, DFW, "divergence-free wavelet soft-thresholding"),
		OPT_UINT('j', &flags, "bitmask", "joint soft-thresholding"),
		OPT_INT('b', &llrblk, "blocksize", "locally low rank block size"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int N = DIMS;
	long dims[N];
	complex float* idata = load_cfl(argv[2], N, dims);
	complex float* odata = create_cfl(argv[3], N, dims);

	float lambda = atof(argv[1]);

	switch (th_type) {

		case WAV:
			wthresh(N, dims, lambda, flags, odata, idata);
			break;

		case LLR:
			lrthresh(N, dims, llrblk, lambda, flags, odata, idata);
			break;
                        
		case DFW:
			dfthresh(N, dims, lambda, odata, idata);
			break;

		case HARD:
			hard_thresh(N, dims, lambda, odata, idata);
			break;

		default:
			md_zsoftthresh(N, dims, lambda, flags, odata, idata);

	}


	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	exit(0);
}


