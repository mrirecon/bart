/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noir/recon.h"





static const char usage_str[] = "<kspace> <output> [<sensitivities>]";
static const char help_str[] =
		"Jointly estimate image and sensitivities with nonlinear\n"
		"inversion using {iter} iteration steps. Optionally outputs\n"
		"the sensitivities.";



int main_nlinv(int argc, char* argv[])
{
	double start_time = timestamp();

	bool normalize = true;
	bool combine = true;
	unsigned int nmaps = 1;
	float restrict_fov = -1.;
	const char* psf = NULL;
	const char* init_file = NULL;
	struct noir_conf_s conf = noir_defaults;
	bool out_sens = false;
	bool scale_im = false;
	bool usegpu = false;

	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_FLOAT('M', &conf.alpha_min, "", "(minimum for regularization)"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_SET('c', &conf.rvc, "Real-value constraint"),
		OPT_CLEAR('N', &normalize, "Do not normalize image with coil sensitivities"),
		OPT_UINT('m', &nmaps, "nmaps", "Number of ENLIVE maps to use in reconsctruction"),
		OPT_CLEAR('U', &combine, "Do not combine ENLIVE maps in output"),
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_STRING('p', &psf, "PSF", ""),
		OPT_STRING('I', &init_file, "file", "File for initialization"),
		OPT_SET('g', &usegpu, "use gpu"),
		OPT_SET('S', &scale_im, "Re-scale image after reconstruction"),
		OPT_FLOAT('a', &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('b', &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPT_SET('P', &conf.pattern_for_each_coil, "(supplied psf is different for each coil)"),
	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (4 == argc)
		out_sens = true;



	num_init();

	long ksp_dims[DIMS];
	complex float* kspace_data = load_cfl(argv[1], DIMS, ksp_dims);

	// FIXME: SMS should not be the default

	if (1 != ksp_dims[SLICE_DIM]) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		fftmod(DIMS, ksp_dims, SLICE_FLAG, kspace_data, kspace_data); // fftmod to get correct slice order in output
		conf.sms = true;
	}

	// The only multimap we understand with is the one we do ourselves, where
	// we allow multiple images and sensitivities during the reconsctruction
	assert(1 == ksp_dims[MAPS_DIM]);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

	long sens_dims[DIMS];
	md_copy_dims(DIMS, sens_dims, ksp_dims);
	sens_dims[MAPS_DIM] = nmaps;

	long sens_strs[DIMS];
	md_calc_strides(DIMS, sens_strs, sens_dims, CFL_SIZE);


	long img_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|SLICE_FLAG, img_dims, sens_dims);

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	long img_output_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG, img_output_dims, sens_dims);
	if (!combine)
		img_output_dims[MAPS_DIM] = nmaps;

	long img_output_strs[DIMS];
	md_calc_strides(DIMS, img_output_strs, img_output_dims, CFL_SIZE);

	complex float* img_output = create_cfl(argv[2], DIMS, img_output_dims);
	md_clear(DIMS, img_output_dims, img_output, CFL_SIZE);


	complex float* img = md_alloc(DIMS, img_dims, CFL_SIZE);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);

	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = NULL;

	complex float* ksens = md_alloc(DIMS, sens_dims, CFL_SIZE);
	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[3] : "", DIMS, sens_dims);

	// initialization
	if (NULL != init_file) {

		long skip = md_calc_size(DIMS, img_dims);
		long init_dims[DIMS];
		complex float* init = load_cfl(init_file, DIMS, init_dims);

		assert(md_check_bounds(DIMS, 0, img_dims, init_dims));

		md_copy(DIMS, img_dims, img, init, CFL_SIZE);
		fftmod(DIMS, sens_dims, FFT_FLAGS|SLICE_FLAG, ksens, init + skip);

		unmap_cfl(DIMS, init_dims, init);

	} else {

		md_zfill(DIMS, img_dims, img, 1.);
		md_clear(DIMS, sens_dims, ksens, CFL_SIZE);
	}

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != psf) {

		complex float* tmp_psf =load_cfl(psf, DIMS, pat_dims);
		pattern = anon_cfl("", DIMS, pat_dims);

		md_copy(DIMS, pat_dims, pattern, tmp_psf, CFL_SIZE);
		unmap_cfl(DIMS, pat_dims, tmp_psf);
		// FIXME: check compatibility

		if (conf.pattern_for_each_coil) {
			assert( 1 != pat_dims[COIL_DIM] );
		} else {
			if (-1 == restrict_fov)
				restrict_fov = 0.5;

			conf.noncart = true;
		}

	} else {

		md_copy_dims(DIMS, pat_dims, img_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);
	}

#if 0
	float scaling = 1. / estimate_scaling(ksp_dims, NULL, kspace_data);
#else
	double scaling = 100. / md_znorm(DIMS, ksp_dims, kspace_data);

	if (1 != ksp_dims[SLICE_DIM]) // SMS
			scaling *= sqrt(ksp_dims[SLICE_DIM]); 

#endif
	debug_printf(DP_INFO, "Scaling: %f\n", scaling);
	md_zsmul(DIMS, ksp_dims, kspace_data, kspace_data, scaling);

	if (-1. == restrict_fov) {

		mask = md_alloc(DIMS, msk_dims, CFL_SIZE);
		md_zfill(DIMS, msk_dims, mask, 1.);

	} else {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;
		mask = compute_mask(DIMS, msk_dims, restrict_dims);
	}

	complex float* ref = NULL;

#ifdef  USE_CUDA
	if (usegpu) {

		complex float* kspace_gpu = md_alloc_gpu(DIMS, ksp_dims, CFL_SIZE);
		md_copy(DIMS, ksp_dims, kspace_gpu, kspace_data, CFL_SIZE);

		noir_recon(&conf, sens_dims, img, sens, ksens, ref, pattern, mask, kspace_gpu);
		md_free(kspace_gpu);
	} else
#endif
		noir_recon(&conf, sens_dims, img, sens, ksens, ref, pattern, mask, kspace_data);


	// image output
	if (normalize) {

		complex float* buf = md_alloc(DIMS, sens_dims, CFL_SIZE);
		md_clear(DIMS, sens_dims, buf, CFL_SIZE);

		if (combine) {

			md_zfmac2(DIMS, sens_dims, ksp_strs, buf, img_strs, img, sens_strs, sens);
			md_zrss(DIMS, ksp_dims, COIL_FLAG, img_output, buf);
		} else {

			md_zfmac2(DIMS, sens_dims, sens_strs, buf, img_strs, img, sens_strs, sens);
			md_zrss(DIMS, sens_dims, COIL_FLAG, img_output, buf);
		}
		md_zmul2(DIMS, img_output_dims, img_output_strs, img_output, img_output_strs, img_output, msk_strs, mask);

		if (1 == nmaps || !combine) {

			//restore phase
			md_zphsr(DIMS, img_output_dims, buf, img);
			md_zmul(DIMS, img_output_dims, img_output, img_output, buf);
		}

		md_free(buf);
	} else {

		if (combine) {

			// just sum up the map images
			md_zaxpy2(DIMS, img_dims, img_output_strs, img_output, 1., img_strs, img);
		} else { /*!normalize && !combine */

			// Just copy
			md_copy(DIMS, img_output_dims, img_output, img, CFL_SIZE);
		}
	}

	if (scale_im)
		md_zsmul(DIMS, img_output_dims, img_output, img_output, 1. / scaling);


	md_free(mask);
	md_free(img);

	unmap_cfl(DIMS, sens_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, img_output_dims, img_output);
	unmap_cfl(DIMS, ksp_dims, kspace_data);

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	exit(0);
}


