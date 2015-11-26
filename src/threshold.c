/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2015 Frank Ong <frankong@berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <getopt.h>

#include "num/flpmath.h"
#include "num/multind.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#include "wavelet2/wavelet.h"

#include "lowrank/lrthresh.h"

#include "dfwavelet/prox_dfwavelet.h"

// FIXME: lowrank interface should not be coupled to mri.h -- it should take D as an input
#ifndef DIMS
#define DIMS 16
#endif



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-W|L|D] [-b LLR_blksize] [-j bitmask] lambda <input> <output>\n", name);
}

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


int main_threshold(int argc, char* argv[])
{
	unsigned int flags = 0;
        
	enum th_type { NONE, WAV, LLR, DFW, MPDFW } th_type = NONE;
	int llrblk = 8;

	char c;
	while (-1 != (c = getopt(argc, argv, "WLDb:j:h"))) {

		switch (c) {

		case 'j':
			flags = atoi(optarg);
			break;

		case 'W':
			th_type = WAV;
			break;

		case 'L':
			th_type = LLR;
			break;
                        
		case 'D':
			th_type = DFW;
			break;
                        
		case 'b':
			llrblk = atoi(optarg);
			break;

		case 'h':
			usage(argv[0], stdout);
			printf(	"\nPerform soft-thresholding with parameter lambda.\n\n"
                                "-W\t\tdaubechies wavelet soft-thresholding\n"
                                "-L\t\tlocally low rank soft-thresholding\n"
                                "-D\t\tdivergence-free wavelet soft-thresholding\n"
				"-j bitmask\tjoint soft-thresholding\n"
				"-h\t\thelp\n"		);
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 3) {

		usage(argv[0], stderr);
		exit(1);
	}

	const int N = DIMS;
	long dims[N];
	complex float* idata = load_cfl(argv[optind + 1], N, dims);
	complex float* odata = create_cfl(argv[optind + 2], N, dims);

	float lambda = atof(argv[optind + 0]);

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


