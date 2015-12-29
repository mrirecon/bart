/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015. Martin Uecker
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2013, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2015 Frank Ong <frankong@berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "wavelet2/wavelet.h"

#include "lowrank/lrthresh.h"

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

	const struct linop_s* w = wavelet_create(D, dims, 7, minsize, false, false);
	const struct operator_p_s* p = prox_unithresh_create(D, w, lambda, flags, false);

	operator_p_apply(p, 1., D, dims, out, D, dims, in);

	operator_p_free(p);
}

static void lrthresh(unsigned int D, const long dims[D], int llrblk, float lambda, unsigned int flags, complex float* out, const complex float* in)
{

	long blkdims[MAX_LEV][D];

	int levels = llr_blkdims(blkdims, ~flags, dims, llrblk);
	UNUSED(levels);

	// FIXME: problem with randshift = false?
	const struct operator_p_s* p = lrthresh_create(dims, true, ~flags, (const long (*)[])blkdims, lambda, false, false, false);

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

        assert(dims[TE_DIM]==3);

        const struct operator_p_s* p = prox_dfwavelet_create( dims, minsize, res, TE_DIM, lambda, false );

	operator_p_apply(p, 1., D, dims, out, D, dims, in);

	operator_p_free(p);
}



static const char* usage_str = "lambda <input> <output>";
static const char* help_str = "Perform soft-thresholding with parameter lambda.";



int main_threshold(int argc, char* argv[])
{
	unsigned int flags = 0;
        
	enum th_type { NONE, WAV, LLR, DFW, MPDFW } th_type = NONE;
	int llrblk = 8;


	const struct opt_s opts[] = {

		{ 'W', false, opt_select, OPT_SEL(enum th_type, &th_type, WAV), "\t\tdaubechies wavelet soft-thresholding" },
		{ 'L', false, opt_select, OPT_SEL(enum th_type, &th_type, LLR), "\t\tlocally low rank soft-thresholding" },
		{ 'D', false, opt_select, OPT_SEL(enum th_type, &th_type, DFW), "\t\tdivergence-free wavelet soft-thresholding" },
		{ 'j', true, opt_int, &flags, " bitmask\tjoint soft-thresholding" },
		{ 'b', true, opt_int, &llrblk, NULL },
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);


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

		default:
			md_zsoftthresh(N, dims, lambda, flags, odata, idata);

	}


	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	exit(0);
}


