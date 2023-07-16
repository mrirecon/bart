/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Frank Ong <uecker@eecs.berkeley.edu>
 *
 *
 * Ra JB, Rim CY. Fast imaging using subencoding data sets from multiple detectors. 
 * Magn Reson Med 1993; 30:142-145.
 *
 * Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast
 * MRI. Magn Reson Med 1999; 42:952-962.
 *
 * Pruessmann KP, Weiger M, Boernert P, Boesiger P. Advances in sensitivity
 * encoding with arbitrary k-space trajectories. 
 * Magn Reson Med 2001; 46:638-651.
 *
 * Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M.
 * ESPIRiT - An Eigenvalue Approach to  Autocalibrating Parallel MRI: Where SENSE 
 * meets GRAPPA. Magn Reson Med 2014; 71:990-1001.
 *
 */

#include <string.h>
#include <complex.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif
#include "num/multiplace.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"


#include "model.h"


struct linop_s* linop_sampling_create(const long dims[DIMS], const long pat_dims[DIMS], const complex float* pattern)
{
	assert(md_check_compat(DIMS, ~0UL, dims, pat_dims));

	auto ret = (struct linop_s*)linop_cdiag_create(DIMS, dims, md_nontriv_dims(DIMS, pat_dims), NULL);
	linop_gdiag_set_diag_ref(ret, DIMS, pat_dims, pattern);

	return ret;
}


/**
 * Create maps operator, m = S x
 *
 * @param max_dims maximal dimensions across all data structures
 * @param sens_flags active map dimensions
 * @param sens sensitivities
 */
struct linop_s* maps_create(const long max_dims[DIMS], 
			unsigned int sens_flags, const complex float* sens)
{
	long mps_dims[DIMS];
	md_select_dims(DIMS, sens_flags, mps_dims, max_dims);
	complex float* nsens = md_alloc_sameplace(DIMS, mps_dims, CFL_SIZE, sens);
	fftscale(DIMS, mps_dims, FFT_FLAGS, nsens, sens);

	long cim_dims[DIMS];
	long img_dims[DIMS];

	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, max_dims);
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

	auto ret = (struct linop_s*)linop_fmac_dims_create(DIMS, cim_dims, img_dims, mps_dims, NULL);
	linop_fmac_set_tensor_F(ret, DIMS, mps_dims, nsens);

	return ret;
}



struct linop_s* maps2_create(const long coilim_dims[DIMS], const long maps_dims[DIMS], const long img_dims[DIMS], const complex float* maps)
{
	unsigned long sens_flags = 0;

	for (unsigned int i = 0; i < DIMS; i++)
		if (1 != maps_dims[i])
			sens_flags = MD_SET(sens_flags, i);

	assert(1 == coilim_dims[MAPS_DIM]);
	assert(1 == img_dims[COIL_DIM]);
	assert(maps_dims[COIL_DIM] == coilim_dims[COIL_DIM]);
	assert(maps_dims[MAPS_DIM] == img_dims[MAPS_DIM]);

	auto ret = (struct linop_s*)linop_fmac_dims_create(DIMS, coilim_dims, img_dims, maps_dims, NULL);
	linop_fmac_set_tensor_ref(ret, DIMS, maps_dims, maps);

	return ret;
}




/**
 * Create sense operator, y = F S x,
 * where F is the Fourier transform and S is the sensitivity maps
 *
 * @param max_dims maximal dimensions across all data structures
 * @param sens_flags active map dimensions
 * @param sens sensitivities
 */
struct linop_s* sense_init(const long max_dims[DIMS], 
			unsigned int sens_flags, const complex float* sens)
{
	long ksp_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, ksp_dims, max_dims);

	struct linop_s* fft = linop_fft_create(DIMS, ksp_dims, FFT_FLAGS);
	struct linop_s* maps = maps_create(max_dims, sens_flags, sens);

	struct linop_s* sense_op = linop_chain(maps, fft);

	linop_free(fft);
	linop_free(maps);

	return sense_op;
}



