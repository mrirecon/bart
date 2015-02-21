/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a educational/research license which can be found in the
 * LICENSE file.
 *
 * Authors:
 * 2012-2014, Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "calib/calmat.h"


static void usage(const char* name, FILE* fd)
{
	fprintf(fd, 	"Usage: %s [-k kernel_size] [-r cal_size]"
			" <kspace> <calibration matrix>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Compute calibration matrix.\n"
		"\n"
		"-k ksize\tkernel size\n"
		"-r cal_size\tLimits the size of the calibration region.\n");
}




int main_calmat(int argc, char* argv[])
{
	long calsize[3] = { 24, 24, 24 };
	long kdims[3] = { 5, 5, 5 };
	bool calcen = false;

	int c;
	while (-1 != (c = getopt(argc, argv, "Ck:K:r:R:h"))) {

		switch (c) {

		case 'k':

			kdims[0] = atoi(optarg);
			kdims[1] = atoi(optarg);
			kdims[2] = atoi(optarg);
			break;

		case 'K':
			sscanf(optarg, "%ld:%ld:%ld", &kdims[0], &kdims[1], &kdims[2]);
			break;

		case 'r':
			calsize[0] = atoi(optarg);
			calsize[1] = atoi(optarg);
			calsize[2] = atoi(optarg);
			break;

		case 'R':
			sscanf(optarg, "%ld:%ld:%ld", &calsize[0], &calsize[1], &calsize[2]);
			break;

		case 'C':
			calcen = true;
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

	int N = DIMS;
	long ksp_dims[N];

	complex float* in_data = load_cfl(argv[optind + 0], N, ksp_dims);


	assert(1 == ksp_dims[MAPS_DIM]);



	long cal_dims[N];
	complex float* cal_data = NULL;

	if (!calcen) {

		cal_data = extract_calib(cal_dims, calsize, ksp_dims, in_data, false);

	} else {

		for (int i = 0; i < 3; i++)
			cal_dims[i] = (calsize[i] < ksp_dims[i]) ? calsize[i] : ksp_dims[i];

		for (int i = 3; i < N; i++)
			cal_dims[i] = ksp_dims[i];

		cal_data = md_alloc(N, cal_dims, CFL_SIZE);
		md_resize_center(N, cal_dims, cal_data, ksp_dims, in_data, CFL_SIZE);
	 }

	 for (int i = 0; i < 3; i++)
		 if (1 == ksp_dims[i])
			 kdims[i] = 1;



	 for (unsigned int i = 0; i < 3; i++)
		 if ((1 == cal_dims[i]) && (1 != ksp_dims[i])) {

			fprintf(stderr, "Calibration region not found!\n");
			exit(1);
		}


	// FIXME: we should scale the data

	unmap_cfl(N, ksp_dims, in_data);


	long calmat_dims[N];
	md_singleton_dims(N, calmat_dims);
	complex float* cm = calibration_matrix(calmat_dims, kdims, cal_dims, cal_data);
	md_free(cal_data);

	complex float* out_data = create_cfl(argv[optind + 1], N, calmat_dims);
	md_copy(N, calmat_dims, out_data, cm, CFL_SIZE);
	md_free(cm);

	unmap_cfl(N, calmat_dims, out_data);

	exit(0);
}


