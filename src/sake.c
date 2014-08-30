/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 *
 * Peter J. Shin, Peder E.Z. Larson, Michael A. Ohliger, Michael Elad,
 * John M. Pauly, Daniel B. Vigneron and Michael Lustig, Calibrationless
 * Parallel Imaging Reconstruction Based on Structured Low-Rank Matrix 
 * Completion, 2013, accepted to Magn Reson Med.
 *
 * Zhongyuan Bi, Martin Uecker, Dengrong Jiang, Michael Lustig, and Kui Ying.
 * Robust Low-rank Matrix Completion for sparse motion correction in auto 
 * calibration PI. Annual Meeting ISMRM, Salt Lake City 2013, 
 * In Proc. Intl. Soc. Mag. Recon. Med 21; 2584 (2013)
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <getopt.h>

#include "num/init.h"
#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "sake/sake.h"

#undef DIMS // FIXME
#define DIMS 5

// 8 channels: alpha 0.2, 50 iter

static void usage(const char* name, FILE* fd)
{
        fprintf(fd, "Usage: %s [-i iterations] [-s rel. subspace] <kspace> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Use SAKE algorithm to recover a full k-space from undersampled\n"
		"data using low-rank matrix completion.\n"
		"\n"
		"-i\tnumber of iterations\n"
		"-s\trel. size of the signal subspace\n");
}


int main(int argc, char* argv[])
{
	int c;
	float alpha = 0.22;
	int iter = 50;
	float lambda = 1.;

	while (-1 != (c = getopt(argc, argv, "i:s:o:h"))) {

		switch (c) {

		case 'i':

			iter = atoi(optarg);
			break;	

		case 's':

			alpha = atof(optarg);
			break;

		case 'o':

			lambda = atof(optarg);
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

	assert((0. <= alpha) && (alpha <= 1.));
	assert(iter >= 0);
	assert((0. <= lambda) && (lambda <= 1.));

	long dims[DIMS];

	num_init();
	
	complex float* in_data = load_cfl(argv[optind + 0], DIMS, dims);
	complex float* out_data = create_cfl(argv[optind + 1], DIMS, dims);

	lrmc(alpha, iter, lambda, DIMS, dims, out_data, in_data);

	unmap_cfl(DIMS, dims, out_data);
	unmap_cfl(DIMS, dims, in_data);
	exit(0);
}


