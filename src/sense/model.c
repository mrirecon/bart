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
#include "num/ops.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"


#include "model.h"




/**
 * data structure for holding the sense data.
 *
 * @param max_dims maximal dimensions 
 * @param dims_mps maps dimensions
 * @param dims_ksp kspace dimensions
 * @param img_dims final image dimensions
 * @param strs_mps strides for maps
 * @param strs_ksp strides for kspace
 * @param strs_img strides for image
 * @param sens sensitivity maps
  */
struct maps_data {

	long max_dims[DIMS];

	long mps_dims[DIMS];
	long ksp_dims[DIMS];
	long img_dims[DIMS];

	long strs_mps[DIMS];
	long strs_ksp[DIMS];
	long strs_img[DIMS];

	/*const*/ complex float* sens;
	complex float* norm;
};



static void maps_apply(const void* _data, complex float* dst, const complex float* src)
{
	const struct maps_data* data = _data;

	md_clear(DIMS, data->ksp_dims, dst, CFL_SIZE);
	md_zfmac2(DIMS, data->max_dims, data->strs_ksp, dst, data->strs_img, src, data->strs_mps, data->sens);
}


static void maps_apply_adjoint(const void* _data, complex float* dst, const complex float* src)
{
 	const struct maps_data* data = _data;

	// dst = sum( conj(sens) .* tmp )
	md_clear(DIMS, data->img_dims, dst, CFL_SIZE);
	md_zfmacc2(DIMS, data->max_dims, data->strs_img, dst, data->strs_ksp, src, data->strs_mps, data->sens);
}


static void maps_init_normal(struct maps_data* data)
{
	if (NULL != data->norm)
		return;

	data->norm = md_alloc_sameplace(DIMS, data->img_dims, CFL_SIZE, data->sens);
	md_zrss(DIMS, data->mps_dims, COIL_FLAG, data->norm, data->sens);
	md_zmul(DIMS, data->img_dims, data->norm, data->norm, data->norm);
}


static void maps_apply_normal(const void* _data, complex float* dst, const complex float* src)
{
	struct maps_data* data = (struct maps_data*)_data;

	maps_init_normal(data);

	md_zmul(DIMS, data->img_dims, dst, src, data->norm);
}


/*
 * ( AT A + lambda I) x = b
 */
static void maps_apply_pinverse(const void* _data, float lambda, complex float* dst, const complex float* src)
{
	struct maps_data* data = (struct maps_data*)_data;

	maps_init_normal(data);

	md_zsadd(DIMS, data->img_dims, data->norm, data->norm, lambda);
	md_zdiv(DIMS, data->img_dims, dst, src, data->norm);
	md_zsadd(DIMS, data->img_dims, data->norm, data->norm, -lambda);
}


static void maps_free_data(const void* _data)
{
	const struct maps_data* data = _data;
	md_free((void*)data->sens);
	free((void*)data);
}


static struct maps_data* maps_create_data(const long max_dims[DIMS], 
			unsigned int sens_flags, const complex float* sens, bool gpu)
{
	PTR_ALLOC(struct maps_data, data);

	// maximal dimensions
	md_copy_dims(DIMS, data->max_dims, max_dims);

	// sensitivity dimensions
	md_select_dims(DIMS, sens_flags, data->mps_dims, max_dims);
	md_calc_strides(DIMS, data->strs_mps, data->mps_dims, CFL_SIZE);

	md_select_dims(DIMS, ~MAPS_FLAG, data->ksp_dims, max_dims);
	md_calc_strides(DIMS, data->strs_ksp, data->ksp_dims, CFL_SIZE);

	md_select_dims(DIMS, ~COIL_FLAG, data->img_dims, max_dims);
	md_calc_strides(DIMS, data->strs_img, data->img_dims, CFL_SIZE);

	
#ifdef USE_CUDA
	complex float* nsens = (gpu ? md_alloc_gpu : md_alloc)(DIMS, data->mps_dims, CFL_SIZE);
#else
	assert(!gpu);
	complex float* nsens = md_alloc(DIMS, data->mps_dims, CFL_SIZE);
#endif
	md_copy(DIMS, data->mps_dims, nsens, sens, CFL_SIZE);
	data->sens = nsens;

	data->norm = NULL;

	return data;
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
	struct maps_data* data = maps_create_data(max_dims, sens_flags, sens, gpu);

	// scale the sensitivity maps by the FFT scale factor
	fftscale(DIMS, data->mps_dims, FFT_FLAGS, data->sens, data->sens);

	return linop_create(DIMS, data->ksp_dims, DIMS, data->img_dims, data, 
			maps_apply, maps_apply_adjoint, maps_apply_normal, maps_apply_pinverse, maps_free_data);
}



struct linop_s* maps2_create(const long coilim_dims[DIMS], const long maps_dims[DIMS], const long img_dims[DIMS], const complex float* maps, bool use_gpu)
{
	long max_dims[DIMS];

	unsigned int sens_flags = 0;

	for (unsigned int i = 0; i < DIMS; i++)
		if (1 != maps_dims[i])
			sens_flags = MD_SET(sens_flags, i);

	assert(1 == coilim_dims[MAPS_DIM]);
	assert(1 == img_dims[COIL_DIM]);
	assert(maps_dims[COIL_DIM] == coilim_dims[COIL_DIM]);
	assert(maps_dims[MAPS_DIM] == img_dims[MAPS_DIM]);

	for (unsigned int i = 0; i < DIMS; i++)
		max_dims[i] = MAX(coilim_dims[i], MAX(maps_dims[i], img_dims[i]));

	struct maps_data* data = maps_create_data(max_dims, sens_flags, maps, use_gpu);

	return linop_create(DIMS, coilim_dims, DIMS, img_dims, data,
		maps_apply, maps_apply_adjoint, maps_apply_normal, maps_apply_pinverse, maps_free_data);
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
	long ksp_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, ksp_dims, max_dims);

	struct linop_s* fft = linop_fft_create(DIMS, ksp_dims, FFT_FLAGS, gpu);
	struct linop_s* maps = maps_create(max_dims, sens_flags, sens, gpu);

	struct linop_s* sense_op = linop_chain(maps, fft);

	linop_free(fft);
	linop_free(maps);

	return sense_op;
}



