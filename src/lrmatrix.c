/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a educational/research license which can be found in the 
 * LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu>
 */


#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/ops.h"

#include "linops/linop.h"

#include "iter/iter.h"
#include "iter/lsqr.h"
#include "iter/thresh.h"

#include "lowrank/lrthresh.h"
#include "linops/sum.h"
#include "linops/sampling.h"
#include "iter/prox.h"
#include "linops/someops.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"

struct s_data{
	long size;
};

// x = (z1 + z2)/2

static void sum_xupdate( const void* _data, float rho, complex float* dst, const complex float* src )
{
	UNUSED(rho);

	const struct s_data* data = (const struct s_data*) _data;

	for( int i = 0; i < data->size; i++)
		dst[i] = src[i] / 2.;
}

static void sum_xupdate_free( const void* data )
{
	free( (void*) data);
}



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-options] <input> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Perform (multi-scale) low rank matrix completion\n"
                "-d\t\tperform decomposition instead, ie fully sampled\n"
                "-i\t\tmaximum iterations.\n"
                "-m\t\tflags to specify which dimensions are reshaped to matrix columns.\n"
                "-f\t\tflags to specify which dimensions to perform multi-scale partition.\n"
                "-j scale\tblock size scaling from one scale to the next one.\n"
                "-k block-size\tsmallest block size\n"
                "-N\t\tadd noise scale to account for Gaussian noise.\n"
                "-s\t\tperform low rank + sparse matrix completion.\n"
                "-l block-size\tperform locally low rank soft thresholding with specified block size.\n"
                "-o <output2>\tsummed over all non-noise scales to create a denoised output.\n"
		"\n");
}


int main_lrmatrix(int argc, char* argv[])
{
	double start_time = timestamp();

	bool use_gpu = false;

	int maxiter = 100;
	float rho = 0.25;
	int blkskip = 2;
	_Bool randshift = true;
	unsigned long mflags = 1;
	unsigned long flags = ~0;
	const char* sum_str = NULL;
	_Bool noise = false;
        _Bool decom = false;

	_Bool llr = false;
	long llrblk = 8;
	_Bool ls = false;
	_Bool hogwild = false;
	_Bool fast = true;
	long initblk = 1;
	int remove_mean = 0;

	int c;
	while (-1 != (c = getopt(argc, argv, "uvNi:p:m:j:k:o:hnl:sf:gHFd"))) {
		switch(c) {

                case 'd':
                        decom = true;
                        
		case 'u':
                        remove_mean = 1;
			break;

		case 'v':
			remove_mean = 2;
			break;

		case 'H':
			hogwild = true;
			break;

		case 'k':
			initblk = atoi(optarg);
			break;

		case 'o':
			sum_str = strdup(optarg);
			break;


		case 'i':
			maxiter = atoi(optarg);
			break;

		case 'j':
			blkskip = atoi(optarg);
			break;

		case 'p':
			rho = atof(optarg);
			break;

		case 'N':
			noise = true;
			break;

		case 'n':
			randshift = false;
			break;

		case 'l':
			llr = true;
			llrblk = atoi(optarg);
			break;

		case 's':
			ls = true;
			break;

		case 'f':
			flags = labs(atol(optarg));
			break;

		case 'm':
			mflags = labs(atol(optarg));
			break;

		case 'g':
			use_gpu = true;
			break;

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 2) {

		usage(argv[0], stderr);
		exit(1);
	}


	long idims[DIMS];
	long odims[DIMS];

	// Load input
	complex float* idata = load_cfl(argv[optind + 0], DIMS, idims);

	// Get levels and block dimensions
	long blkdims[MAX_LEV][DIMS];
	long levels;
	if (llr)
		levels = llr_blkdims(blkdims, flags, idims, llrblk);
	else if (ls)
		levels = ls_blkdims(blkdims, idims);
	else
		levels = multilr_blkdims(blkdims, flags, idims, blkskip, initblk);

	if (noise)
		add_lrnoiseblk( &levels, blkdims, idims );
	debug_printf(DP_INFO, "Number of levels: %ld\n", levels);

	// Get outdims
	md_copy_dims(DIMS, odims, idims);
	odims[LEVEL_DIM] = levels;
	complex float* odata = create_cfl(argv[optind + 1], DIMS, odims);
	md_clear( DIMS, odims, odata, sizeof(complex float) );

	// Get pattern
	complex float* pattern = NULL;

        if (!decom) {
                pattern = md_alloc(DIMS, idims, CFL_SIZE);
                estimate_pattern(DIMS, idims, TIME_DIM, pattern, idata);
        }

	// Initialize algorithm
	void* iconf;

	struct iter_admm_conf mmconf;
	memcpy(&mmconf, &iter_admm_defaults, sizeof(struct iter_admm_conf));
	mmconf.maxiter = maxiter;
	mmconf.rho = rho;
	mmconf.hogwild = hogwild;
	mmconf.fast = fast;
	
	iconf = &mmconf;


	// Initialize operators

	const struct linop_s* sum_op = sum_create( odims, use_gpu );
	const struct linop_s* sampling_op = NULL;
        if (!decom) {
                sampling_op = sampling_create(idims, idims, pattern);
                sum_op = linop_chain(sum_op, sampling_op);
                linop_free(sampling_op);
        }
	
	const struct operator_p_s* sum_prox = prox_lineq_create( sum_op, idata );
	const struct operator_p_s* lr_prox = lrthresh_create(odims, randshift, mflags, (const long (*)[])blkdims, 1., noise, remove_mean, use_gpu);

        assert(use_gpu == false);

	(use_gpu ? num_init_gpu : num_init)();

	if (use_gpu)
		debug_printf(DP_INFO, "GPU reconstruction\n");

	// put into iter2 format
	unsigned int num_funs = 2;
	const struct linop_s* eye_op = linop_identity_create(DIMS, odims);
	const struct linop_s* ops[2] = { eye_op, eye_op };
	const struct operator_p_s* prox_ops[2] = { sum_prox, lr_prox };
	long size = 2 * md_calc_size(DIMS, odims);
	struct s_data s_data = { size / 2 };

	const struct operator_p_s* sum_xupdate_op = operator_p_create( DIMS, odims, DIMS, odims, (void*) &s_data, sum_xupdate, sum_xupdate_free );


	// do recon
	
	iter2_admm( iconf,
		    NULL,
		    num_funs,
		    prox_ops,
		    ops,
		    sum_xupdate_op,
		    size, (float*) odata, NULL,
		    NULL, NULL, NULL );
	


	// Sum
	if (sum_str)
	{
		complex float* sdata = create_cfl(sum_str, DIMS, idims);
		long istrs[DIMS];
		long ostrs[DIMS];

		md_calc_strides(DIMS, istrs, idims, sizeof(complex float));
		md_calc_strides(DIMS, ostrs, odims, sizeof(complex float));

		md_clear(DIMS, idims, sdata, sizeof(complex float));
		odims[LEVEL_DIM]--;
		md_zaxpy2(DIMS, odims, istrs, sdata, 1./sqrt(levels), ostrs, odata);
		odims[LEVEL_DIM]++;
		unmap_cfl(DIMS, idims, sdata);
	}


	// Clean up
	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(DIMS, odims, odata);
	linop_free( sum_op );
	operator_p_free( sum_prox );
	operator_p_free( lr_prox );


	double end_time = timestamp();
	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);
	exit(0);
}

