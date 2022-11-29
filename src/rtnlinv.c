/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2020 Christian Holme, <christian.holme@med.uni-goettingen.de>
 * 2018-2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
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
 * Uecker M, Zhang S, Voit D, Karaus A, Merboldt KD, Frahm J.
 * Real-time MRI at a resolution of 20 ms. NMR Biomed 2010; 23:986-994.
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

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/version.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noir/recon.h"
#include "noir/misc.h"

#include "noncart/nufft.h"

#include "linops/linop.h"





static const char help_str[] =
		"Jointly estimate a time-series of images and sensitivities with nonlinear\n"
		"inversion using {iter} iteration steps. Optionally outputs\n"
		"the sensitivities.";





int main_rtnlinv(int argc, char* argv[argc])
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

	struct nufft_conf_s nufft_conf = nufft_conf_defaults;
	nufft_conf.toeplitz = false;

	bool normalize = true;
	bool combine = true;
	unsigned int nmaps = 1;
	float restrict_fov = -1.;
	float temp_damp = 0.9f;
	const char* psf = NULL;
	const char* trajectory = NULL;
	const char* init_file = NULL;
	const char* init_file_im = NULL;
	struct noir_conf_s conf = noir_defaults;
	bool out_sens = false;
	bool scale_im = false;
	bool use_gpu = false;
	float scaling = -1.;
	bool alt_scaling = false;


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
		OPT_INFILE('p', &psf, "file", "pattern / transfer function"),
		OPT_INFILE('t', &trajectory, "file", "kspace trajectory"),
		OPT_INFILE('I', &init_file, "file", "File for initialization"),
		OPT_INFILE('C', &init_file_im, "", "(File for initialization with image space sensitivities)"),
		OPT_SET('g', &use_gpu, "use gpu"),
		OPT_SET('S', &scale_im, "Re-scale image after reconstruction"),
		OPT_FLOAT('a', &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('b', &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('T', &temp_damp, "temp_damp", "temporal damping [default: 0.9]"),
		OPT_FLOAT('w', &scaling, "", "(inverse scaling of the data)"),
		OPT_VEC3('x', &my_img_dims, "x:y:z", "Explicitly specify image dimensions"),
		OPT_SET('A', &alt_scaling, "(Alternative scaling)"), // Used for SSA-FARY paper
 		OPT_SET('s', &conf.sms, "(Simultaneous Multi-Slice reconstruction)")
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (NULL != sens_file)
		out_sens = true;

	(use_gpu ? num_init_gpu : num_init)();

	if ((NULL != psf) && (NULL != trajectory))	// FIXME: pattern makes sense with trajectory
		error("Pass either trajectory (-t) or PSF (-p)!\n");


	long ksp_dims[DIMS];
	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

	long frames = ksp_dims[TIME_DIM];

	long ksp1_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, ksp1_dims, ksp_dims);


	// SMS
	if (conf.sms) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		fftmod(DIMS, ksp_dims, SLICE_FLAG, kspace, kspace); // fftmod to get correct slice order in output
	}

	long pat_dims[DIMS];
	complex float* pattern = NULL;

	long trj_dims[DIMS];
	long trj1_dims[DIMS];
	complex float* traj = NULL;
	int turns = 1;

	long sens_dims[DIMS];
	md_copy_dims(DIMS, sens_dims, ksp_dims);

	sens_dims[MAPS_DIM] = nmaps;


	if (NULL != psf) {

		conf.noncart = true;

		// copy here so that pattern is only ever a pointer allocated by md_alloc
		complex float* tmp_pattern = load_cfl(psf, DIMS, pat_dims);

		pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
		md_copy(DIMS, pat_dims, pattern, tmp_pattern, CFL_SIZE);

		unmap_cfl(DIMS, pat_dims, tmp_pattern);

		turns = pat_dims[TIME_DIM];

	} else if (NULL != trajectory) {

		conf.noncart = true;

		traj = load_cfl(trajectory, DIMS, trj_dims);

		turns = trj_dims[TIME_DIM];

		md_select_dims(DIMS, ~TIME_FLAG, trj1_dims, trj_dims);

		//debug_print_dims(DP_INFO, 3, my_img_dims);

		if (!alt_scaling)
			md_zsmul(DIMS, trj_dims, traj, traj, 2.);

		if (0 == my_img_dims[0] + my_img_dims[1] + my_img_dims[2])
			estimate_fast_sq_im_dims(3, sens_dims, trj_dims, traj);
		else
			md_copy_dims(3, sens_dims, my_img_dims);

	} else {

		error("Pass either trajectory (-t) or PSF (-p)!\n");
	}


	long sens1_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, sens1_dims, sens_dims);



	if (-1 == restrict_fov)
		restrict_fov = 0.5;


	// The only multimap we understand with is the one we do ourselves, where
	// we allow multiple images and sensitivities during the reconsctruction (ENLIVE)
	assert(1 == ksp_dims[MAPS_DIM]);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, sens_dims);

	long img1_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, img1_dims, img_dims);

	long img_output_dims[DIMS];
	md_copy_dims(DIMS, img_output_dims, img_dims);

	if (conf.noncart && !alt_scaling) {

		for (int i = 0; i < 3; i++)
			if (1 != img_output_dims[i])
				img_output_dims[i] /= 2;
	}

	long img_output1_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, img_output1_dims, img_output_dims);

	if (combine) {

		// The conventional img-dimensions contain only one map.
		// The 'output' dimensions might contain multiple maps (ENLIVE)
		img_output_dims[MAPS_DIM] = 1;
		img_output1_dims[MAPS_DIM] = 1;
	}

	complex float* img_output = create_cfl(img_file, DIMS, img_output_dims);
	md_clear(DIMS, img_output_dims, img_output, CFL_SIZE);

	complex float* img1 = md_alloc(DIMS, img1_dims, CFL_SIZE);


	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, img1_dims);

	complex float* mask = NULL;

	// Full output sensitivities
	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? sens_file : "", DIMS, sens_dims);

	// Sensitivities in image domain (relevant for output and normalization)
	complex float* sens1 = md_calloc(DIMS, sens1_dims, CFL_SIZE);

	// Sensitivities in kspace-domain (relevant for reference and initialization)
	complex float* ksens1 = md_alloc(DIMS, sens1_dims, CFL_SIZE);
	md_clear(DIMS, sens1_dims, ksens1, CFL_SIZE);

	long skip = md_calc_size(DIMS, img1_dims);
	long size = skip + md_calc_size(DIMS, sens1_dims);

	// initialization
	if (NULL != init_file) {

		long init_dims[DIMS];
		complex float* init = load_cfl(init_file, DIMS, init_dims);

		if (!md_check_bounds(DIMS, 0, img1_dims, init_dims))
			error("Image dimensions and init dimensions do not match!");

		md_copy(DIMS, img1_dims, img1, init, CFL_SIZE);
		md_clear(DIMS, sens1_dims, ksens1, CFL_SIZE);

		unmap_cfl(DIMS, init_dims, init);

	} else if (NULL != init_file_im) {
		
		long init_dims[DIMS];
		complex float* init = load_cfl(init_file_im, DIMS, init_dims);

		if (!md_check_bounds(DIMS, 0, img1_dims, init_dims))
			error("Image dimensions and init dimensions do not match!");

		md_copy(DIMS, img1_dims, img1, init, CFL_SIZE);
		md_copy(DIMS, sens1_dims, ksens1, init + skip, CFL_SIZE);

		conf.img_space_coils = true;

		unmap_cfl(DIMS, init_dims, init);

	} else {

		md_zfill(DIMS, img1_dims, img1, 1.);
		md_clear(DIMS, sens1_dims, ksens1, CFL_SIZE);
	}



	// Gridding
	if (NULL != trajectory) {

		debug_printf(DP_DEBUG3, "Start gridding psf ...");

		md_select_dims(DIMS, ~(COIL_FLAG|MAPS_FLAG), pat_dims, sens_dims);
		pat_dims[TIME_DIM] = turns;


		long wgh_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, wgh_dims, ksp_dims);
		complex float* wgh = md_alloc(DIMS, wgh_dims, CFL_SIZE);

		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, wgh, kspace);
		pattern = compute_psf(DIMS, pat_dims, trj_dims, traj, trj_dims, NULL, wgh_dims, wgh, false, false);

		md_free(wgh);


		fftuc(DIMS, pat_dims, FFT_FLAGS, pattern, pattern);

		if (alt_scaling) {

			if (frames > turns) {

				// For turn-based reconstructions we use the conventional scaling
				float patnorm = 1.f / ksp_dims[2];
				md_zsmul(DIMS, pat_dims, pattern, pattern, patnorm);

			} else {

				// This scaling accounts for variable spokes per frame
				scale_psf_k(pat_dims, pattern, ksp_dims, kspace, trj_dims, traj);
			}

		} else {

			float psf_sc = 1.;

			for (int i = 0; i < 3; i++)
				if (1 != pat_dims[i])
					psf_sc *= 2.;

			md_zsmul(DIMS, pat_dims, pattern, pattern, psf_sc);
		}

		debug_printf(DP_DEBUG3, "finished\n");
	}


	long kgrid_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, kgrid_dims, sens_dims);

	long kgrid1_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, kgrid1_dims, kgrid_dims);


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

	long ref_dim[1] = { size };
	complex float* ref = md_calloc(1, ref_dim, CFL_SIZE);

	if (NULL != init_file_im) { // Prepare refrence from init file
		
		md_zsmul(DIMS, img1_dims, ref, img1, temp_damp);
		md_zsmul(DIMS, sens1_dims, ref + skip, ksens1, temp_damp);
	}

	struct linop_s* nufft_ops[turns];
	const struct operator_s* fftc = NULL;

	complex float* fftc_mod = NULL;
	complex float* kgrid1 = NULL;
	complex float* traj1 = NULL;

	if (NULL != trajectory) { 	// Crecte nufft objects

		debug_printf(DP_DEBUG3, "Start creating nufft-objects...");

		traj1 = md_alloc(DIMS, trj1_dims, CFL_SIZE);

		for (int i = 0; i < turns; ++i) {

			// pick trajectory for current frame
			long pos[DIMS] = { 0 };
			pos[TIME_DIM] = i;
			md_slice(DIMS, TIME_FLAG, pos, trj_dims, traj1, traj, CFL_SIZE);

			nufft_ops[i] = nufft_create(DIMS, ksp1_dims, kgrid1_dims, trj1_dims, traj1, NULL, nufft_conf);
		}

		kgrid1 = md_alloc(DIMS, kgrid1_dims, CFL_SIZE);

		fftc = fft_measure_create(DIMS, kgrid1_dims, FFT_FLAGS, true, false);
		fftc_mod = md_alloc(DIMS, kgrid1_dims, CFL_SIZE);

		md_zfill(DIMS, kgrid1_dims, fftc_mod, 1.);
		fftmod(DIMS, kgrid1_dims, FFT_FLAGS, fftc_mod, fftc_mod);

		debug_printf(DP_DEBUG3, "finished\n");
	}


	debug_printf(DP_DEBUG3, "Start reconstruction\n");

	complex float* img_output1 = md_alloc(DIMS, img_output1_dims, CFL_SIZE);
	complex float* sens_output1 = md_alloc(DIMS, sens1_dims, CFL_SIZE);
	complex float* kspace1 = md_alloc(DIMS, ksp1_dims, CFL_SIZE);

	long pat1_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, pat1_dims, pat_dims);

	complex float* pattern1 = md_alloc(DIMS, pat1_dims, CFL_SIZE);

	for (int frame = 0; frame < frames; ++frame) {

		debug_printf(DP_DEBUG1, "Reconstructing frame %d\n", frame);

		// pick k-space and pattern for current frame
		long pos[DIMS] = { 0 };
		pos[TIME_DIM] = frame;

		md_slice(DIMS, TIME_FLAG, pos, ksp_dims, kspace1, kspace, CFL_SIZE);

		pos[TIME_DIM] = frame % turns;
		md_slice(DIMS, TIME_FLAG, pos, pat_dims, pattern1, pattern, CFL_SIZE);

		if (NULL != trajectory)  {

			// grid data frame by frame
			linop_adjoint(nufft_ops[frame % turns], DIMS, kgrid1_dims, kgrid1, DIMS, ksp1_dims, kspace1);
#if 1
			md_zmul(DIMS, kgrid1_dims, kgrid1, kgrid1, fftc_mod);
			fft_exec(fftc, kgrid1, kgrid1);
			md_zmul(DIMS, kgrid1_dims, kgrid1, kgrid1, fftc_mod);
			fftscale(DIMS, kgrid1_dims, FFT_FLAGS, kgrid1, kgrid1);
#else
			fftuc(DIMS, kgrid1_dims, FFT_FLAGS, kgrid1, kgrid1);
#endif
			if (!use_compat_to_version("v0.7.00")) {

				float sc = 1.;
				for (int i = 0; i < 3; i++)
					if (1 != kgrid1_dims[i])
						sc *= 2.;

				md_zsmul(DIMS, kgrid1_dims, kgrid1, kgrid1, sqrtf(sc));
			}
		} else {

			kgrid1 = kspace1;
		}


		if ((-1. == scaling) || alt_scaling)
			scaling = 100. / md_znorm(DIMS, kgrid1_dims, kgrid1);

		md_zsmul(DIMS, kgrid1_dims, kgrid1, kgrid1, scaling);

#ifdef USE_CUDA
		if (use_gpu) {

			complex float* kgrid1_gpu = md_alloc_gpu(DIMS, kgrid1_dims, CFL_SIZE);
			md_copy(DIMS, kgrid1_dims, kgrid1_gpu, kgrid1, CFL_SIZE);

			noir_recon(&conf, sens1_dims, img1, sens1, ksens1, ref, pattern1, mask, kgrid1_gpu);
			md_free(kgrid1_gpu);

		} else
#endif
			noir_recon(&conf, sens1_dims, img1, sens1, ksens1, ref, pattern1, mask, kgrid1);


		// Temporal regularization
		md_zsmul(DIMS, img1_dims, ref, img1, temp_damp);
		md_zsmul(DIMS, sens1_dims, ref + skip, ksens1, temp_damp);

		long img_output1_strs[DIMS];
		md_calc_strides(DIMS, img_output1_strs, img_output1_dims, CFL_SIZE);

		long img1_strs[DIMS];
		md_calc_strides(DIMS, img1_strs, img1_dims, CFL_SIZE);

		long sens1_strs[DIMS];
		md_calc_strides(DIMS, sens1_strs, sens1_dims, CFL_SIZE);


		postprocess(sens1_dims, normalize,
				sens1_strs, sens1,
				img1_strs, img1,
				img_output1_dims, img_output1_strs, img_output1);

		if (scale_im)
			md_zsmul(DIMS, img_output1_dims, img_output1, img_output1, 1. / scaling);

		// Copy frame to correct position in output array
		long pos2[DIMS] = { 0 };
		pos2[TIME_DIM] = frame;

		md_copy_block(DIMS, pos2, img_output_dims, img_output, img_output1_dims, img_output1, CFL_SIZE);

		if (out_sens)
			md_copy_block(DIMS, pos2, sens_dims, sens, sens1_dims, sens1, CFL_SIZE);

		if (NULL != init_file_im)
			conf.img_space_coils = false;
	}

	md_free(mask);
	md_free(img1);
	md_free(kspace1);
	md_free(pattern1);
	md_free(sens_output1);
	md_free(img_output1);
	md_free(sens1);
	md_free(ksens1);
	md_free(ref);
	md_free(pattern);

	if (NULL != trajectory) {

		md_free(traj1);
		md_free(kgrid1);
		md_free(fftc_mod);

		unmap_cfl(DIMS, trj_dims, traj);

		operator_free(fftc);

		for (int i = 0; i < turns; ++i)
			linop_free(nufft_ops[i]);
	}

	unmap_cfl(DIMS, sens_dims, sens);
	unmap_cfl(DIMS, img_output_dims, img_output);
	unmap_cfl(DIMS, ksp_dims, kspace);


	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	return 0;
}


