/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2014 Martin Uecker <uecker@eecs.berkeley.edu>
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
#include "num/someops.h"
#include "num/linop.h"
#include "num/ops.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"


#include "model.h"




/**
 * data structure for holding the sense data.
 *
 * @param dims_max maximal dimensions 
 * @param dims_mps maps dimensions
 * @param dims_ksp kspace dimensions
 * @param dims_img final image dimensions
 * @param strs_mps strides for maps
 * @param strs_ksp strides for kspace
 * @param strs_img strides for image
 * @param sens sensitivity maps
  */
struct maps_data {

	long dims_max[DIMS];

	long dims_mps[DIMS];
	long dims_ksp[DIMS];
	long dims_img[DIMS];

	long strs_mps[DIMS];
	long strs_ksp[DIMS];
	long strs_img[DIMS];

	const complex float* sens;
};



static void maps_apply(const void* _data, complex float* dst, const complex float* src)
{
	const struct maps_data* data = _data;

	md_clear(DIMS, data->dims_ksp, dst, CFL_SIZE);
	md_zfmac2(DIMS, data->dims_max, data->strs_ksp, dst, data->strs_img, src, data->strs_mps, data->sens);
}



static void maps_apply_adjoint(const void* _data, complex float* dst, const complex float* src)
{
 	const struct maps_data* data = _data;

	// dst = sum( conj(sens) .* tmp )
	md_clear(DIMS, data->dims_img, dst, CFL_SIZE);
	md_zfmacc2(DIMS, data->dims_max, data->strs_img, dst, data->strs_ksp, src, data->strs_mps, data->sens);
}




static void maps_free(const void* _data)
{
	const struct maps_data* data = _data;
	md_free((void*)data->sens);
	free((void*)data);
}

/**
 * Create maps operator, m = S x
 *
 * @param max_dims maximal dimensions across all data structures
 * @param sens_flags active map dimensions
 * @param sens sensitivities
 * @param gpu TRUE if using gpu
 */
struct linop_s* maps_create(const long max_dims[DIMS], 
			unsigned int sens_flags, const complex float* sens, bool gpu)
{
	struct maps_data* data = xmalloc(sizeof(struct maps_data));

	// maximal dimensions
	md_copy_dims(DIMS, data->dims_max, max_dims);

	// sensitivity dimensions
	md_select_dims(DIMS, sens_flags, data->dims_mps, max_dims);
	md_calc_strides(DIMS, data->strs_mps, data->dims_mps, CFL_SIZE);

	// kspace dimensions include TE_DIM
	md_select_dims(DIMS, ~MAPS_FLAG, data->dims_ksp, max_dims);
	md_calc_strides(DIMS, data->strs_ksp, data->dims_ksp, CFL_SIZE);

	// image dimensions include TE_DIM
	md_select_dims(DIMS, ~COIL_FLAG, data->dims_img, max_dims);
	md_calc_strides(DIMS, data->strs_img, data->dims_img, CFL_SIZE);

	
	// scale the sensitivity maps by the FFT scale factor
#ifdef USE_CUDA
	complex float* nsens = (gpu ? md_alloc_gpu : md_alloc)(DIMS, data->dims_mps, CFL_SIZE);
#else
	assert(!gpu);
	complex float* nsens = md_alloc(DIMS, data->dims_mps, CFL_SIZE);
#endif
	fftscale(DIMS, data->dims_mps, FFT_FLAGS, nsens, sens);
	data->sens = nsens;

	return linop_create(DIMS, data->dims_ksp, data->dims_img, data, maps_apply, maps_apply_adjoint, NULL, NULL, maps_free);
}





/**
 * Create sense operator, y = F S x,
 * where F is the Fourier transform and S is the sensitivity maps
 *
 * @param max_dims maximal dimensions across all data structures
 * @param sens_flags active map dimensions
 * @param sens sensitivities
 * @param gpu TRUE if using gpu
 */
struct linop_s* sense_init(const long max_dims[DIMS], 
			unsigned int sens_flags, const complex float* sens, bool gpu)
{
	long dims_ksp[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, dims_ksp, max_dims);

	struct linop_s* fft = linop_fft_create(DIMS, dims_ksp, FFT_FLAGS, gpu);
	struct linop_s* maps = maps_create(max_dims, sens_flags, sens, gpu);

	struct linop_s* sense_op = linop_chain(maps, fft);

	linop_free(fft);
	linop_free(maps);

	return sense_op;
}





// FIXME get rid of wrappers?

void sense_free(const struct linop_s* o)
{
	linop_free(o);
}




/**
 * Wrapper for sense_apply
 */
void sense_forward(const struct linop_s* o, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(o, dst, src);
}



void sense_adjoint(const struct linop_s* o, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(o, dst, src);
}




