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
	bool normalize = true;
	float restrict_fov = -1.;
	const char* psf = NULL;
	struct noir_conf_s conf = noir_defaults;

	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", ""),
		OPT_SET('c', &conf.rvc, ""),
		OPT_CLEAR('N', &normalize, ""),
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_STRING('p', &psf, "PSF", ""),
		OPT_SET('g', &conf.usegpu, "use gpu"),
	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long ksp_dims[DIMS];
	complex float* kspace_data = load_cfl(argv[1], DIMS, ksp_dims);

	long dims[DIMS];
	md_copy_dims(DIMS, dims, ksp_dims);


	long img_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|CSHIFT_FLAG, img_dims, dims);

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);


	complex float* image = create_cfl(argv[2], DIMS, img_dims);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, dims);

	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask; 
	complex float* norm = md_alloc(DIMS, msk_dims, CFL_SIZE);
	complex float* sens = ((4 == argc) ? create_cfl : anon_cfl)((4 == argc) ? argv[3] : "", DIMS, ksp_dims);


	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != psf) {

		pattern = load_cfl(psf, DIMS, pat_dims);

		// FIXME: check compatibility
	} else {

		md_copy_dims(DIMS, pat_dims, img_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);
	}

#if 0
	float scaling = 1. / estimate_scaling(ksp_dims, NULL, kspace_data);
#else
	float scaling = 100. / md_znorm(DIMS, ksp_dims, kspace_data);
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

#ifdef  USE_CUDA
	if (conf.usegpu) {

		complex float* kspace_gpu = md_alloc_gpu(DIMS, ksp_dims, CFL_SIZE);
		md_copy(DIMS, ksp_dims, kspace_gpu, kspace_data, CFL_SIZE);
		noir_recon(&conf, dims, image, NULL, pattern, mask, kspace_gpu);
		md_free(kspace_gpu);

		md_zfill(DIMS, ksp_dims, sens, 1.);

	} else
#endif
	noir_recon(&conf, dims, image, sens, pattern, mask, kspace_data);

	if (normalize) {

		md_zrss(DIMS, ksp_dims, COIL_FLAG, norm, sens);
		md_zmul2(DIMS, img_dims, img_strs, image, img_strs, image, msk_strs, norm);
	}

	if (4 == argc) {

		long strs[DIMS];

		md_calc_strides(DIMS, strs, ksp_dims, CFL_SIZE);

		if (norm)
			md_zdiv2(DIMS, ksp_dims, strs, sens, strs, sens, img_strs, norm);

		fftmod(DIMS, ksp_dims, FFT_FLAGS, sens, sens);
	}


	md_free(norm);
	md_free(mask);

	unmap_cfl(DIMS, ksp_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, img_dims, image);
	unmap_cfl(DIMS, ksp_dims, kspace_data);
	exit(0);
}


