/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/filter.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noncart/nufft.h"
#include "linops/linop.h"

#include "moba/recon_T1.h"




static const char usage_str[] = "<kspace> <TI/TE> <output> [<sensitivities>]";
static const char help_str[] = "Model-based nonlinear inverse reconstruction\n";


int main_moba(int argc, char* argv[])
{
	double start_time = timestamp();

	float restrict_fov = -1.;
	float oversampling = 1.25f;
	unsigned int sample_size = 0;
	unsigned int grid_size = 0;
	const char* psf = NULL;
	const char* trajectory = NULL;
	struct moba_conf conf = moba_defaults;
	bool out_sens = false;
	bool usegpu = false;
	bool unused = false;
	enum mdb_t { MDB_T1 } mode = { MDB_T1 };

	const struct opt_s opts[] = {

		OPT_SELECT('L', enum mdb_t, &mode, MDB_T1, "T1 mapping using model-based look-locker"),
		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPT_FLOAT('j', &conf.alpha_min, "", "Minimum regu. parameter"),
		OPT_UINT('C', &conf.inner_iter, "iter", "inner iterations"),
		OPT_FLOAT('s', &conf.step, "step", "step size"),
		OPT_FLOAT('B', &conf.lower_bound, "bound", "lower bound for relaxivity"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_SET('N', &unused, "(normalize)"), // no-op
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_STRING('p', &psf, "PSF", ""),
		OPT_SET('M', &conf.sms, "Simultaneous Multi-Slice reconstruction"),
		OPT_SET('g', &usegpu, "use gpu"),
		OPT_STRING('t', &trajectory, "Traj", ""),
		OPT_FLOAT('o', &oversampling, "os", "Oversampling factor for gridding [default: 1.25]"),
		OPT_SET('k', &conf.k_filter, "k-space edge filter for non-Cartesian trajectories"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (5 == argc)
		out_sens = true;


	num_init();

	long ksp_dims[DIMS];
	complex float* kspace_data = load_cfl(argv[1], DIMS, ksp_dims);

	long TI_dims[DIMS];
	complex float* TI = load_cfl(argv[2], DIMS, TI_dims);

	assert(TI_dims[TE_DIM] == ksp_dims[TE_DIM]);
	assert(1 == ksp_dims[MAPS_DIM]);

	if (conf.sms) {

		debug_printf(DP_INFO, "SMS Model-based reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		fftmod(DIMS, ksp_dims, SLICE_FLAG, kspace_data, kspace_data); // fftmod to get correct slice order in output
	}

	long grid_dims[DIMS];
	md_copy_dims(DIMS, grid_dims, ksp_dims);

	if (NULL != trajectory) {

		sample_size = ksp_dims[1];
		grid_size = sample_size * oversampling;
		grid_dims[READ_DIM] = grid_size;
		grid_dims[PHS1_DIM] = grid_size;
		grid_dims[PHS2_DIM] = 1L;
				
		if (-1 == restrict_fov)
			restrict_fov = 0.5;

		conf.noncartesian = true;
	}

	long img_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|COEFF_FLAG|SLICE_FLAG|TIME2_FLAG, img_dims, grid_dims);

	img_dims[COEFF_DIM] = 3;

	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	long single_map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|SLICE_FLAG|TIME2_FLAG, single_map_dims, grid_dims);

	long single_map_strs[DIMS];
	md_calc_strides(DIMS, single_map_strs, single_map_dims, CFL_SIZE);

	long coil_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|MAPS_FLAG|SLICE_FLAG|TIME2_FLAG, coil_dims, grid_dims);

	long coil_strs[DIMS];
	md_calc_strides(DIMS, coil_strs, coil_dims, CFL_SIZE);

	complex float* img = create_cfl(argv[3], DIMS, img_dims);
	complex float* single_map = anon_cfl("", DIMS, single_map_dims);

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, grid_dims);

	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = NULL;
	complex float* norm = md_alloc(DIMS, img_dims, CFL_SIZE);
	complex float* sens = (out_sens ? create_cfl : anon_cfl)(out_sens ? argv[4] : "", DIMS, coil_dims);


	md_zfill(DIMS, img_dims, img, 1.0);
	md_clear(DIMS, coil_dims, sens, CFL_SIZE);

	complex float* k_grid_data = NULL;
	k_grid_data = anon_cfl("", DIMS, grid_dims);

	complex float* pattern = NULL;
	long pat_dims[DIMS];
	

	if (NULL != psf) {

		complex float* tmp_psf =load_cfl(psf, DIMS, pat_dims);
		pattern = anon_cfl("", DIMS, pat_dims);

		md_copy(DIMS, pat_dims, pattern, tmp_psf, CFL_SIZE);
		unmap_cfl(DIMS, pat_dims, tmp_psf);

		md_copy(DIMS, grid_dims, k_grid_data, kspace_data, CFL_SIZE);
		unmap_cfl(DIMS, ksp_dims, kspace_data);

		if (0 == md_check_compat(DIMS, COIL_FLAG, ksp_dims, pat_dims))
			error("pattern not compatible with kspace dimensions\n");

		if (-1 == restrict_fov)
			restrict_fov = 0.5;

		conf.noncartesian = true;

	} else if (NULL != trajectory) {

		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = false;

		struct linop_s* nufft_op_p = NULL;
		struct linop_s* nufft_op_k = NULL;

		long traj_dims[DIMS];
		long traj_strs[DIMS];

		complex float* traj = load_cfl(trajectory, DIMS, traj_dims);
		md_calc_strides(DIMS, traj_strs, traj_dims, CFL_SIZE);

		md_zsmul(DIMS, traj_dims, traj, traj, oversampling);

		long ones_dims[DIMS];
		md_copy_dims(DIMS, ones_dims, traj_dims);
		ones_dims[READ_DIM] = 1L;
		complex float* ones = md_alloc(DIMS, ones_dims, CFL_SIZE);
		md_zfill(DIMS, ones_dims, ones, 1.0);

		// Gridding sampling pattern
		
		md_select_dims(DIMS, FFT_FLAGS|TE_FLAG|SLICE_FLAG|TIME2_FLAG, pat_dims, grid_dims);
		pattern = anon_cfl("", DIMS, pat_dims);

		nufft_op_p = nufft_create(DIMS, ones_dims, pat_dims, traj_dims, traj, NULL, nufft_conf);
		linop_adjoint(nufft_op_p, DIMS, pat_dims, pattern, DIMS, ones_dims, ones);
		fftuc(DIMS, pat_dims, FFT_FLAGS, pattern, pattern);

		// Gridding raw data

		nufft_op_k = nufft_create(DIMS, ksp_dims, grid_dims, traj_dims, traj, NULL, nufft_conf);
		linop_adjoint(nufft_op_k, DIMS, grid_dims, k_grid_data, DIMS, ksp_dims, kspace_data);
		fftuc(DIMS, grid_dims, FFT_FLAGS, k_grid_data, k_grid_data);

		linop_free(nufft_op_p);
		linop_free(nufft_op_k);

		md_free(ones);
		unmap_cfl(DIMS, ksp_dims, kspace_data);

	} else {

		md_copy_dims(DIMS, pat_dims, img_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);
		md_copy(DIMS, grid_dims, k_grid_data, kspace_data, CFL_SIZE);
		unmap_cfl(DIMS, ksp_dims, kspace_data);
	}


	if (conf.k_filter) {

		long map_dims[DIMS];
		md_select_dims(DIMS, FFT_FLAGS, map_dims, pat_dims);

		long map_strs[DIMS];
		md_calc_strides(DIMS, map_strs, map_dims, CFL_SIZE);

		long pat_strs[DIMS];
		md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

		complex float* filter = NULL;
		filter = anon_cfl("", DIMS, map_dims);
		float lambda = 2e-3;

		klaplace(DIMS, map_dims, READ_FLAG|PHS1_FLAG, filter);
		md_zreal(DIMS, map_dims, filter, filter);
		md_zsqrt(DIMS, map_dims, filter, filter);

		md_zsmul(DIMS, map_dims, filter, filter, -2.);
		md_zsadd(DIMS, map_dims, filter, filter, 1.);
		md_zatanr(DIMS, map_dims, filter, filter);

		md_zsmul(DIMS, map_dims, filter, filter, -1. / M_PI);
		md_zsadd(DIMS, map_dims, filter, filter, 1.0);

		md_zsmul(DIMS, map_dims, filter, filter, lambda);

		md_zadd2(DIMS, pat_dims, pat_strs, pattern, pat_strs, pattern, map_strs, filter);

		unmap_cfl(DIMS, map_dims, filter);
	}

	double scaling = 5000. / md_znorm(DIMS, grid_dims, k_grid_data);
	double scaling_psf = 1000. / md_znorm(DIMS, pat_dims, pattern);

        if (conf.sms) {

		scaling *= grid_dims[SLICE_DIM] / 5.0;
		scaling_psf *= grid_dims[SLICE_DIM] / 5.0;
	}

	debug_printf(DP_INFO, "Scaling: %f\n", scaling);
	md_zsmul(DIMS, grid_dims, k_grid_data, k_grid_data, scaling);

	debug_printf(DP_INFO, "Scaling_psf: %f\n", scaling_psf);
	md_zsmul(DIMS, pat_dims, pattern, pattern, scaling_psf);

	if (-1. == restrict_fov) {

		mask = md_alloc(DIMS, msk_dims, CFL_SIZE);
		md_zfill(DIMS, msk_dims, mask, 1.);

	} else {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;
		mask = compute_mask(DIMS, msk_dims, restrict_dims);
		//md_zsmul2(DIMS, img_dims, img_strs, img, msk_strs, mask ,1.0);

		md_zmul2(DIMS, img_dims, img_strs, img, img_strs, img, msk_strs, mask);

		// Choose a different initial guess for R1*
		long pos[DIMS];

		for (int i = 0; i < (int)DIMS; i++)
			pos[i] = 0;

		pos[COEFF_DIM] = 2;
		md_copy_block(DIMS, pos, single_map_dims, single_map, img_dims, img, CFL_SIZE);
		md_zsmul2(DIMS, single_map_dims, single_map_strs, single_map, single_map_strs, single_map, conf.sms ? 2.0 : 1.5);
		md_copy_block(DIMS, pos, img_dims, img, single_map_dims, single_map, CFL_SIZE);
	}

#ifdef  USE_CUDA
	if (usegpu) {

//		cuda_use_global_memory();

		complex float* kspace_gpu = md_alloc_gpu(DIMS, grid_dims, CFL_SIZE);
		md_copy(DIMS, grid_dims, kspace_gpu, k_grid_data, CFL_SIZE);

		complex float* TI_gpu = md_alloc_gpu(DIMS, TI_dims, CFL_SIZE);
		md_copy(DIMS, TI_dims, TI_gpu, TI, CFL_SIZE);

		switch (mode) {

		case MDB_T1:
			T1_recon(&conf, grid_dims, img, sens, pattern, mask, TI_gpu, kspace_gpu, usegpu);
			break;
		};

		md_free(kspace_gpu);
		md_free(TI_gpu);
	} else
#endif
	switch (mode) {

	case MDB_T1:
		T1_recon(&conf, grid_dims, img, sens, pattern, mask, TI, k_grid_data, usegpu);
		break;
	};

	md_free(norm);
	md_free(mask);

	unmap_cfl(DIMS, coil_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, grid_dims, k_grid_data);
	unmap_cfl(DIMS, img_dims, img);
	unmap_cfl(DIMS, single_map_dims, single_map);
	unmap_cfl(DIMS, TI_dims, TI);

	double recosecs = timestamp() - start_time;
	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);
	exit(0);
}

