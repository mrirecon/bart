/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * Publications:
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction
 * by regularized nonlinear inversion-joint estimation of coil
 * sensitivities and image content. Magn Reson Med 2008; 60:674-682.
 *
 * Uecker M, Zhang S, Frahm J. Nonlinear Inverse Reconstruction for
 * Real-time MRI of the Human Heart Using Undersampled Radial FLASH.
 * Magn Reson Med 2010; 63:1456-1462.
 *
 * Holme HCM, Rosenzweig S, Ong F, Wilke RN, Lustig M, Uecker M.
 * ENLIVE: An Efficient Nonlinear Method for Calibrationless and
 * Robust Parallel Imaging. Sci Rep 2019; 9:3034.
 *
 * Rosenzweig S, Holme HMC, Wilke RN, Voit D, Frahm J, Uecker M.
 * Simultaneous multi-slice MRI using cartesian and radial FLASH and
 * regularized nonlinear inversion: SMS-NLINV.
 * Magn Reson Med 2018; 79:2057--2066.
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "noncart/nufft.h"
#include "linops/linop.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/version.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noir/recon.h"
#include "noir/misc.h"





static const char help_str[] =
		"Jointly estimate image and sensitivities with nonlinear\n"
		"inversion using {iter} iteration steps. Optionally outputs\n"
		"the sensitivities.";



int main_nlinv(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* ksp_file = NULL;
	const char* img_file = NULL;
	const char* sens_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_OUTFILE(true, &img_file, "output"),
		ARG_OUTFILE(false, &sens_file, "sensitivities"),
	};

	bool normalize = true;
	bool combine = true;
	unsigned int nmaps = 1;
	float restrict_fov = -1.;
	const char* psf_file = NULL;
	const char* trajectory = NULL;
	const char* init_file = NULL;
	struct noir_conf_s conf = noir_defaults;
	bool scale_im = false;
	bool use_gpu = false;
	float scaling = -1.;
	bool nufft_lowmem = false;

	long my_img_dims[3] = { 0, 0, 0 };

	const struct opt_s opts[] = {

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_FLOAT('M', &conf.alpha_min, "", "(minimum for regularization)"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_SET('c', &conf.rvc, "Real-value constraint"),
		OPT_CLEAR('N', &normalize, "Do not normalize image with coil sensitivities"),
		OPT_UINT('m', &nmaps, "nmaps", "Number of ENLIVE maps to use in reconstruction"),
		OPT_CLEAR('U', &combine, "Do not combine ENLIVE maps in output"),
		OPT_FLOAT('f', &restrict_fov, "FOV", "restrict FOV"),
		OPT_INFILE('p', &psf_file, "file", "pattern / transfer function"),
		OPT_INFILE('t', &trajectory, "file", "kspace trajectory"),
		OPT_INFILE('I', &init_file, "file", "File for initialization"),
		OPT_SET('g', &use_gpu, "use gpu"),
		OPT_SET('S', &scale_im, "Re-scale image after reconstruction"),
		OPT_UINT('s', &conf.cnstcoil_flags, "", "(dimensions with constant sensitivities)"),
		OPT_FLOAT('a', &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('b', &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPT_SET('P', &conf.pattern_for_each_coil, "(supplied psf is different for each coil)"),
		OPT_SET('n', &conf.noncart, "(non-Cartesian)"),
		OPT_FLOAT('w', &scaling, "", "(inverse scaling of the data)"),
		OPTL_SET(0, "lowmem", &nufft_lowmem, "Use low-mem mode of the nuFFT"),
		OPT_VEC3('x', &my_img_dims, "x:y:z", "Explicitly specify image dimensions"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);


	(use_gpu ? num_init_gpu : num_init)();

	long ksp_dims[DIMS];
	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

	// FIXME: SMS should not be the default

	if (1 != ksp_dims[SLICE_DIM]) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		fftmod(DIMS, ksp_dims, SLICE_FLAG, kspace, kspace); // fftmod to get correct slice order in output
		conf.sms = true;
	}

	// The only multimap we understand with is the one we do ourselves, where
	// we allow multiple images and sensitivities during the reconstruction
	assert(1 == ksp_dims[MAPS_DIM]);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

	long dims[DIMS];
	md_copy_dims(DIMS, dims, ksp_dims);

	complex float* traj = NULL;
	long trj_dims[DIMS];

	if (NULL != trajectory) {
#if 0
		conf.noncart = true;

		traj = load_cfl(trajectory, DIMS, trj_dims);

		estimate_im_dims(DIMS, FFT_FLAGS, dims, trj_dims, traj);
		debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", dims[0], dims[1], dims[2]);

		md_zsmul(DIMS, trj_dims, traj, traj, 2.);

		for (int i = 0; i < DIMS; i++)
			if (MD_IS_SET(FFT_FLAGS, i) && (1 < dims[i]))
				dims[i] *= 2;

		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);
#else
		conf.noncart = true;

		traj = load_cfl(trajectory, DIMS, trj_dims);

		debug_print_dims(DP_INFO, 3, my_img_dims);

		const float oversampling = 2;

		md_zsmul(DIMS, trj_dims, traj, traj, oversampling);

		if (0 == my_img_dims[0] + my_img_dims[1] + my_img_dims[2]) {

			estimate_fast_sq_im_dims(3, dims, trj_dims, traj);
		} else {

			md_copy_dims(3, dims, my_img_dims);

			for (int i = 0; i < 3; i++)
				if (1 != dims[i])
					dims[i] *= oversampling;
		}

#endif




	}	

	// for ENLIVE maps
	dims[MAPS_DIM] = nmaps;

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	long sens_dims[DIMS];
	md_select_dims(DIMS, ~conf.cnstcoil_flags, sens_dims, dims);

	long sens_strs[DIMS];
	md_calc_strides(DIMS, sens_strs, sens_dims, CFL_SIZE);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	long img_output_dims[DIMS];
	md_copy_dims(DIMS, img_output_dims, img_dims);

	if (conf.noncart) {

		for (int i = 0; i < 3; i++)
			if (1 != img_output_dims[i])
				img_output_dims[i] /= 2;
	}

	if (combine)
		img_output_dims[MAPS_DIM] = 1;

	long img_output_strs[DIMS];
	md_calc_strides(DIMS, img_output_strs, img_output_dims, CFL_SIZE);

	complex float* img_output = create_cfl(img_file, DIMS, img_output_dims);
	md_clear(DIMS, img_output_dims, img_output, CFL_SIZE);


	complex float* img = md_alloc(DIMS, img_dims, CFL_SIZE);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img_dims);

	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = NULL;

	complex float* ksens = md_alloc(DIMS, sens_dims, CFL_SIZE);
	complex float* sens = ((NULL != sens_file) ? create_cfl : anon_cfl)((NULL != sens_file) ? sens_file : "", DIMS, sens_dims);

	// initialization
	if (NULL != init_file) {

		long skip = md_calc_size(DIMS, img_dims);
		long init_dims[DIMS];

		complex float* init = load_cfl(init_file, DIMS, init_dims);

		assert(md_check_bounds(DIMS, 0, img_dims, init_dims));

		md_copy(DIMS, img_dims, img, init, CFL_SIZE);
		fftmod(DIMS, sens_dims, FFT_FLAGS | (conf.sms ? SLICE_FLAG : 0u), ksens, init + skip);

		unmap_cfl(DIMS, init_dims, init);

	} else {

		md_zfill(DIMS, img_dims, img, 1.);
		md_clear(DIMS, sens_dims, ksens, CFL_SIZE);
	}


	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != psf_file) {

		pattern = load_cfl(psf_file, DIMS, pat_dims);

		// FIXME: check compatibility

		if (conf.pattern_for_each_coil) {

			assert(sens_dims[COIL_DIM] == pat_dims[COIL_DIM]);
		}

	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);

		pattern = anon_cfl("", DIMS, pat_dims);

		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
	}


	complex float* psf = NULL;
	long psf_dims[DIMS];

	complex float* kgrid = NULL;
	long kgrid_dims[DIMS];
	struct linop_s* nufft_op;


	if ((-1 == restrict_fov) && conf.noncart)
		restrict_fov = 0.5;



	if (NULL != trajectory) {

		debug_printf(DP_DEBUG3, "Start gridding psf ...");

		md_select_dims(DIMS, ~(COIL_FLAG|MAPS_FLAG), psf_dims, sens_dims);

		psf = compute_psf(DIMS, psf_dims, trj_dims, traj, trj_dims, NULL, pat_dims, pattern, false, nufft_lowmem);

		fftuc(DIMS, psf_dims, FFT_FLAGS, psf, psf);

		float psf_sc = 1.;

		for (int i = 0; i < 3; i++)
			if (1 != psf_dims[i])
				psf_sc *= 2.;

		md_zsmul(DIMS, psf_dims, psf, psf, psf_sc);
		debug_printf(DP_DEBUG3, "finished\n");


		debug_printf(DP_DEBUG3, "Start creating nufft-objects...");

		md_select_dims(DIMS, ~MAPS_FLAG, kgrid_dims, sens_dims);

		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = false;
		nufft_conf.lowmem = nufft_lowmem;

		nufft_op = nufft_create(DIMS, ksp_dims, kgrid_dims, trj_dims, traj, NULL, nufft_conf);

		debug_printf(DP_DEBUG3, "finished\n");

		kgrid = anon_cfl("", DIMS, kgrid_dims);

		linop_adjoint(nufft_op, DIMS, kgrid_dims, kgrid, DIMS, ksp_dims, kspace);
		linop_free(nufft_op);

		unmap_cfl(DIMS, ksp_dims, kspace);

		fftuc(DIMS, kgrid_dims, FFT_FLAGS, kgrid, kgrid);

		if (!use_compat_to_version("v0.7.00")) {

			float sc = 1.;
			for (int i = 0; i < 3; i++)
				if (1 != dims[i])
					sc *= 2.;

			md_zsmul(DIMS, kgrid_dims, kgrid, kgrid, sqrtf(sc));
		}

	} else {

		md_copy_dims(DIMS, kgrid_dims, ksp_dims);
		md_copy_dims(DIMS, psf_dims, pat_dims);

		kgrid = kspace;
		psf = pattern;
	}



	if (-1. == scaling) {
#if 0
		scaling = 1. / estimate_scaling(ksp_dims, NULL, kspace);
#else
		scaling = 100. / md_znorm(DIMS, kgrid_dims, kgrid);

		if (conf.sms)
			scaling *= sqrt(kgrid_dims[SLICE_DIM]);
#endif
	}


	debug_printf(DP_INFO, "Scaling: %f\n", scaling);

	md_zsmul(DIMS, kgrid_dims, kgrid, kgrid, scaling);


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
	if (use_gpu) {

		complex float* kspace_gpu = md_alloc_gpu(DIMS, kgrid_dims, CFL_SIZE);

		md_copy(DIMS, kgrid_dims, kspace_gpu, kgrid, CFL_SIZE);

		noir_recon(&conf, dims, img, sens, ksens, ref, psf, mask, kspace_gpu);

		md_free(kspace_gpu);

	} else
#endif
		noir_recon(&conf, dims, img, sens, ksens, ref, psf, mask, kgrid);


	unmap_cfl(DIMS, kgrid_dims, kgrid);

	postprocess(dims, normalize, sens_strs, sens, img_strs, img,
			img_output_dims, img_output_strs, img_output);

	if (scale_im)
		md_zsmul(DIMS, img_output_dims, img_output, img_output, 1. / scaling);


	if (NULL != trajectory)
		md_free(psf);

	md_free(mask);
	md_free(img);
	md_free(ksens);

	unmap_cfl(DIMS, sens_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, img_output_dims, img_output);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	return 0;
}


