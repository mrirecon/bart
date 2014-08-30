/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <getopt.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/fft.h"

#include "calib/walsh.h"


static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s [-r cal. size] [-b block size] <input> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Estimate coil sensitivities using walsh method (use with ecaltwo).\n"
		"\n"
		"-r cal_size\tLimits the size of the calibration region.\n"
		"-b block_size\tBlock size.\n");
}



int main(int argc, char* argv[])
{
	long bsize[3] = { 20, 20, 20 };
	long calsize[3] = { 24, 24, 24 };

	int c;
	while (-1 != (c = getopt(argc, argv, "b:B:r:R:h"))) {

		switch (c) {

		case 'b':

			bsize[0] = atoi(optarg);
			bsize[1] = atoi(optarg);
			bsize[2] = atoi(optarg);
			break;

		case 'B':
			sscanf(optarg, "%ld:%ld:%ld", &bsize[0], &bsize[1], &bsize[2]);
			break;

		case 'r':
			calsize[0] = atoi(optarg);
			calsize[1] = atoi(optarg);
			calsize[2] = atoi(optarg);
			break;

		case 'R':
			sscanf(optarg, "%ld:%ld:%ld", &calsize[0], &calsize[1], &calsize[2]);
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


	long dims[KSPACE_DIMS];

	complex float* in_data = load_cfl(argv[optind + 0], KSPACE_DIMS, dims);

	assert((dims[0] == 1) || (calsize[0] < dims[0]));
	assert((dims[1] == 1) || (calsize[1] < dims[1]));
	assert((dims[2] == 1) || (calsize[2] < dims[2]));
	assert(1 == dims[4]);

	long caldims[KSPACE_DIMS];
	complex float* cal_data = extract_calib(caldims, calsize, dims, in_data, false);
	unmap_cfl(KSPACE_DIMS, dims, in_data);

	printf("Calibration region %ldx%ldx%ld\n", caldims[0], caldims[1], caldims[2]);

	dims[3] = dims[3] * (dims[3] + 1) / 2;
	complex float* out_data = create_cfl(argv[optind + 1], KSPACE_DIMS, dims);

	walsh(bsize, dims, out_data, caldims, cal_data);

	printf("Done.\n");

	md_free(cal_data);
	unmap_cfl(KSPACE_DIMS, dims, out_data);
	exit(0);
}


