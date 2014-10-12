/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
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
#include "num/fft.h"
#include "num/init.h"

#include "grecon/grecon.h"

#include "calib/calib.h"

#include "sense/optcom.h"
#include "sense/recon.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"




static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-l1/-l2] [-r lambda] [-c] <kspace> <sensitivities> <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Perform iterative SENSE/ESPIRiT reconstruction. The read\n"
		"(0th) dimension is Fourier transformed and each section\n"
		"perpendicular to this dimension is reconstructed separately.\n"
		"\n"
		"-l1/-l2\ttoggle l1-wavelet or l2 regularization.\n"
		"-r lambda\tregularization parameter\n"
		"-c\treal-value constraint\n");
}


int main_rsense(int argc, char* argv[])
{
	bool usegpu = false;
	int maps = 2;
	int ctrsh = 0.;
	bool sec = false;

	struct sense_conf sconf;
	memcpy(&sconf, &sense_defaults, sizeof(struct sense_conf));
	struct grecon_conf conf = { SENSE, NULL, &sconf, false, false, false, true, 30, 0.95, 0. };

	int c;
	while (-1 != (c = getopt(argc, argv, "l:r:s:i:q:cgh"))) {

		switch(c) {

		case 'r':
			conf.lambda = atof(optarg);
			break;

		case 's':
			conf.step = atof(optarg);
			break;

		case 'q':
		        conf.sense_conf->cclambda = atof(optarg);
		        break;

		case 'i':
			conf.maxiter = atoi(optarg);
			break;

		case 'l':
			if (1 == atoi(optarg))
				conf.l1wav = true;
			else
			if (2 == atoi(optarg))
				conf.l1wav = false;
			else {
				usage(argv[0], stderr);
				exit(1);
			}
			break;

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		case 'c':
			conf.sense_conf->rvc = true;
			break;

		case 'g':
			usegpu = true;
			break;

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 3) {

		usage(argv[0], stderr);
		exit(1);
	}


	int N = DIMS;
	long dims[N];
	long img_dims[N];
	long ksp_dims[N];
	long sens_dims[N];

	complex float* kspace_data = load_cfl(argv[optind + 0], N, ksp_dims);
	complex float* sens_maps = load_cfl(argv[optind + 1], N, sens_dims);


	assert(1 == ksp_dims[MAPS_DIM]);
//!
	if (!sec) {

		for (int i = 0; i < N; i++)
			dims[i] = sens_dims[i];
	
	} else {

		assert(maps <= ksp_dims[COIL_DIM]);

		for (int i = 0; i < N; i++)
			dims[i] = ksp_dims[i];

		dims[MAPS_DIM] = maps;
	}

#if 1
	for (int i = 0; i < N; i++)
		dims[i] = MAX(ksp_dims[i], sens_dims[i]);
#endif


	// FIXME: higher dimensions?
	for (int i = 0; i < 4; i++) {	// sizes2[4] may be > 1
		if (ksp_dims[i] != dims[i]) {
		
			fprintf(stderr, "Dimensions of kspace and sensitivities do not match!\n");
			exit(1);
		}
	}


	debug_printf(DP_INFO, "%ld map(s)\n", dims[MAPS_DIM]);
	
	if (conf.l1wav)
		debug_printf(DP_INFO, "l1-wavelet regularization\n");


	md_select_dims(N, ~COIL_FLAG, img_dims, dims);

	(usegpu ? num_init_gpu : num_init)();

//	float scaling = estimate_scaling(ksp_dims, sens_maps, kspace_data);
	float scaling = estimate_scaling(ksp_dims, NULL, kspace_data);
	debug_printf(DP_INFO, "Scaling: %f\n", scaling);
	md_zsmul(N, ksp_dims, kspace_data, kspace_data, 1. / scaling);


	debug_printf(DP_INFO, "Readout FFT..\n");
	fftscale(N, ksp_dims, READ_FLAG, kspace_data, kspace_data);
	ifftc(N, ksp_dims, READ_FLAG, kspace_data, kspace_data);
	debug_printf(DP_INFO, "Done.\n");

	complex float* image = create_cfl(argv[optind + 2], N, img_dims);

	debug_printf(DP_INFO, "Reconstruction...\n");

	struct ecalib_conf calib;

	if (sec) {
	
		memcpy(&calib, &ecalib_defaults, sizeof(struct ecalib_conf));
		calib.crop = ctrsh;
		conf.calib = &calib;
	}

	rgrecon(&conf, dims, image, sens_dims, sens_maps, NULL, NULL, kspace_data, usegpu);



	debug_printf(DP_INFO, "Done.\n");

	
	unmap_cfl(N, img_dims, image);
	unmap_cfl(N, ksp_dims, kspace_data);
	unmap_cfl(N, sens_dims, sens_maps);

	exit(0);
}


