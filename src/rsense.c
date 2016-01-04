/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2013, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

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
#include "misc/opts.h"



static const char* usage_str = "<kspace> <sensitivities> <output>";
static const char* help_str =
		"Perform iterative SENSE/ESPIRiT reconstruction. The read\n"
		"(0th) dimension is Fourier transformed and each section\n"
		"perpendicular to this dimension is reconstructed separately.";



int main_rsense(int argc, char* argv[])
{
	bool usegpu = false;
	int maps = 2;
	int ctrsh = 0.;
	bool sec = false;
	bool scale_im = false;
	int lreg = -1;
	const char* pat_file = NULL;

	struct sense_conf sconf = sense_defaults;
	struct grecon_conf conf = { NULL, &sconf, false, false, false, true, 30, 0.95, 0. };

	const struct opt_s opts[] = {

		{ 'l', true, opt_int, &lreg, "1/-l2\t\ttoggle l1-wavelet or l2 regularization." },
		{ 'r', true, opt_float, &conf.lambda, " lambda\tregularization parameter" },
		{ 's', true, opt_float, &conf.step, NULL },
		{ 'i', true, opt_int, &conf.maxiter, NULL },
		{ 'S', false, opt_set, &scale_im, NULL },
		{ 'g', false, opt_set, &usegpu, NULL },
		{ 'p', true, opt_string, &pat_file, NULL },
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);


	if (1 == lreg)
		conf.l1wav = true;
	else
	if (2 == lreg)
		conf.l1wav = false;
	else
		error("Unknown regularization type.");


	int N = DIMS;
	long dims[N];
	long img_dims[N];
	long ksp_dims[N];
	long sens_dims[N];

	complex float* kspace_data = load_cfl(argv[1], N, ksp_dims);
	complex float* sens_maps = load_cfl(argv[2], N, sens_dims);


	assert(1 == ksp_dims[MAPS_DIM]);

	md_copy_dims(N, dims, ksp_dims);

	if (!sec) {

		dims[MAPS_DIM] = sens_dims[MAPS_DIM];
	
	} else {

		assert(maps <= ksp_dims[COIL_DIM]);
		dims[MAPS_DIM] = maps;
	}


	for (int i = 0; i < 4; i++)	// sizes2[4] may be > 1
		if (ksp_dims[i] != dims[i])
			error("Dimensions of kspace and sensitivities do not match!\n");
		

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims);

		assert(md_check_compat(DIMS, COIL_FLAG, ksp_dims, pat_dims));
	}


	debug_printf(DP_INFO, "%ld map(s)\n", dims[MAPS_DIM]);
	
	if (conf.l1wav)
		debug_printf(DP_INFO, "l1-wavelet regularization\n");


	md_select_dims(N, ~COIL_FLAG, img_dims, dims);

	(usegpu ? num_init_gpu : num_init)();

//	float scaling = estimate_scaling(ksp_dims, sens_maps, kspace_data);
	float scaling = estimate_scaling(ksp_dims, NULL, kspace_data);
	debug_printf(DP_INFO, "Scaling: %f\n", scaling);

	if (0. != scaling)
		md_zsmul(N, ksp_dims, kspace_data, kspace_data, 1. / scaling);


	debug_printf(DP_INFO, "Readout FFT..\n");
	ifftuc(N, ksp_dims, READ_FLAG, kspace_data, kspace_data);
	debug_printf(DP_INFO, "Done.\n");

	complex float* image = create_cfl(argv[3], N, img_dims);

	debug_printf(DP_INFO, "Reconstruction...\n");

	struct ecalib_conf calib;

	if (sec) {
	
		calib = ecalib_defaults;
		calib.crop = ctrsh;
		conf.calib = &calib;
	}

	rgrecon(&conf, dims, image, sens_dims, sens_maps, pat_dims, pattern, kspace_data, usegpu);

	if (scale_im)
		md_zsmul(DIMS, img_dims, image, image, scaling);

	debug_printf(DP_INFO, "Done.\n");

	
	unmap_cfl(N, img_dims, image);
	unmap_cfl(N, ksp_dims, kspace_data);
	unmap_cfl(N, sens_dims, sens_maps);

	exit(0);
}


