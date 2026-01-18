/* Copyright 2022-2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <complex.h>
#include <stdbool.h>
#include <stdlib.h>

#include "misc/mri.h"
#include "misc/debug.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"
#include "linops/realval.h"

#include "sense/model.h"
#include "sense/modelnc.h"

#include "motion/displacement.h"

#include "model.h"


const struct linop_s* pics_model(const struct pics_config* conf,
				const long img_dims[DIMS], const long ksp_dims[DIMS],
				const long traj_dims[DIMS], const complex float* traj,
				const long basis_dims[DIMS], const complex float* basis,
				const long map_dims[DIMS], const complex float* maps,
				const long pat_dims[DIMS], const complex float* pattern,
				const long motion_dims[DIMS], complex float* motion,
				const struct linop_s** nufft_op)
{
	const struct linop_s* forward_op = NULL;

	// finalize dimensions

	long max_dims[DIMS];
	md_copy_dims(DIMS, max_dims, ksp_dims);
	md_copy_dims(5, max_dims, map_dims);

	long bmx_dims[DIMS];

	if (NULL != basis) {

		assert(1 == ksp_dims[COEFF_DIM]);

		assert(basis_dims[TE_DIM] == ksp_dims[TE_DIM]);

		max_dims[COEFF_DIM] = basis_dims[COEFF_DIM];

		md_select_dims(DIMS, ~MAPS_FLAG, bmx_dims, max_dims);

		debug_printf(DP_INFO, "Basis: ");
		debug_print_dims(DP_INFO, DIMS, bmx_dims);

		max_dims[TE_DIM] = 1;

		debug_printf(DP_INFO, "Max:   ");
		debug_print_dims(DP_INFO, DIMS, max_dims);
	}

	// make sure the image dimension we get correspond to what we expect

	long img2_dims[DIMS];
	md_select_dims(DIMS, ~(COIL_FLAG | conf->shared_img_flags), img2_dims, max_dims);

	assert(md_check_compat(DIMS, 0UL, img2_dims, img_dims));

	// build model

	if (NULL == traj) {

		unsigned long map_flags = FFT_FLAGS | SENS_FLAGS | md_nontriv_dims(DIMS, map_dims);

		forward_op = sense_init(conf->shared_img_flags & ~conf->motion_flags,
					max_dims, map_flags, maps);

		// apply temporal basis

		if (NULL != basis) {

			const struct linop_s* basis_op = linop_fmac_create(DIMS, bmx_dims, COEFF_FLAG, TE_FLAG, ~(COEFF_FLAG | TE_FLAG), basis);
			forward_op = linop_chain_FF(forward_op, basis_op);
		}

		auto cod = linop_codomain(forward_op);
		const struct linop_s* sample_op = linop_sampling_create(cod->dims, pat_dims, pattern);
		forward_op = linop_chain_FF(forward_op, sample_op);

	} else {

		const complex float* traj_tmp = traj;

		//for computation of psf on GPU
#ifdef USE_CUDA
		if (conf->gpu_gridding) {

			assert(conf->gpu);

			traj_tmp = md_gpu_move(DIMS, traj_dims, traj, sizeof *traj);
		}
#endif

		forward_op = sense_nc_init(max_dims, map_dims, maps, ksp_dims,
				traj_dims, traj_tmp, conf->nuconf,
				pat_dims, pattern, basis_dims, basis, nufft_op,
				conf->shared_img_flags & ~conf->motion_flags);

#ifdef USE_CUDA
		if (conf->gpu_gridding)
			md_free(traj_tmp);
#endif
	}

	if (NULL != motion) {

		long img_motion_dims[DIMS];
		md_copy_dims(DIMS, img_motion_dims, img_dims);
		md_max_dims(DIMS, ~MOTION_FLAG, img_motion_dims, img_motion_dims, motion_dims);

		const struct linop_s* motion_op = linop_interpolate_displacement_create(MOTION_DIM, (1 == img_dims[2]) ? 3 : 7, 1, DIMS, img_motion_dims, motion_dims, motion, img_dims);

		forward_op = linop_chain_FF(motion_op, forward_op);
	}

	if (conf->real_value_constraint)
		forward_op = linop_chain_FF(linop_realval_create(DIMS, img_dims), forward_op);

#ifdef USE_CUDA
	if (conf->gpu && (conf->gpu_gridding || NULL == traj)) {

		auto tmp = linop_gpu_wrapper(forward_op);
		linop_free(forward_op);
		forward_op = tmp;
	}
#endif
	if (conf->time_encoded_asl) {

		const struct linop_s* hadamard_op = linop_hadamard_create(DIMS, img_dims, ITER_DIM);
		forward_op = linop_chain_FF(hadamard_op, forward_op);
	}

	return forward_op;
}



