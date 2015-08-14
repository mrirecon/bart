/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "sense/recon.h"
#include "sense/optcom.h"

#include "misc/mri.h"
#include "misc/mmio.h"




static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s [-h] [-r] <image> <kspace> <sens> <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Recreate k-space from image and sensitivities.\n"
		"\n"
		"-r replace measured samples with original values\n");
}


int main_fakeksp(int argc, char* argv[])
{
	bool rplksp = false;
	int c;

	while (-1 != (c = getopt(argc, argv, "hr"))) {

		switch (c) {

		case 'r':
			rplksp = true;
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

	if (argc - optind != 4) {

		usage(argv[0], stderr);
		exit(1);
	}

	const int N = DIMS;
	long ksp_dims[N];
	long dims[N];
	long img_dims[N];

	complex float* kspace_data = load_cfl(argv[optind + 1], N, ksp_dims);
	complex float* sens_maps = load_cfl(argv[optind + 2], N, dims);
	complex float* image = load_cfl(argv[optind + 0], N, img_dims);
	

	for (int i = 0; i < 4; i++) {

		if (ksp_dims[i] != dims[i]) {
		
			fprintf(stderr, "Dimensions of kspace and sensitivities do not match!\n");
			exit(1);
		}
	}


	assert(1 == ksp_dims[MAPS_DIM]);
	assert(1 == img_dims[COIL_DIM]);
	assert(img_dims[MAPS_DIM] == dims[MAPS_DIM]);

	num_init();

	long dims1[N];

	md_select_dims(N, ~(COIL_FLAG|MAPS_FLAG), dims1, dims);

	long dims2[N];
	md_copy_dims(DIMS, dims2, img_dims);
	dims2[COIL_DIM] = dims[COIL_DIM];
	dims2[MAPS_DIM] = dims[MAPS_DIM];
	


#if 0
	float scaling = estimate_scaling(ksp_dims, NULL, kspace_data);
	printf("Scaling: %f\n", scaling);
	md_zsmul(N, ksp_dims, kspace_data, kspace_data, 1. / scaling);
#endif

	complex float* out = create_cfl(argv[optind + 3], N, ksp_dims);
	
	fftmod(N, ksp_dims, FFT_FLAGS, kspace_data, kspace_data);
	fftmod(N, dims, FFT_FLAGS, sens_maps, sens_maps);

	if (rplksp) {

		printf("Replace kspace\n");
		replace_kspace(dims2, out, kspace_data, sens_maps, image); // this overwrites kspace_data (FIXME: think not!)

	} else {

		printf("Simulate kspace\n");
		fake_kspace(dims2, out, sens_maps, image);
	}

#if 0
	md_zsmul(N, ksp_dims, out, out, scaling);
#endif
	fftmod(N, ksp_dims, FFT_FLAGS, out, out);

	unmap_cfl(N, ksp_dims, kspace_data);
	unmap_cfl(N, dims, sens_maps);
	unmap_cfl(N, img_dims, image);
	unmap_cfl(N, ksp_dims, out);

	exit(0);
}


