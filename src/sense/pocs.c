/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 * Samsonov AA, Kholmovski EG, Parker DL, Johnson CR. POCSENSE: POCS-based
 * reconstruction for sensitivity encoded magnetic resonance imaging.
 * Magn Reson Med 2004; 52:1397–1406.
 *
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/types.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/gpuops.h"
#include "num/ops.h"

#include "linops/linop.h"

#include "iter/iter.h"
#include "iter/prox.h"
#include "iter/monitor.h"

#include "sense/model.h"

#include "pocs.h"



struct data {

	INTERFACE(operator_data_t);

	const struct linop_s* sense_op;
	complex float* tmp;
	float alpha; // l1 or l2 regularization
	float lambda; // robust consistency

	const struct operator_p_s* thresh;

	const complex float* kspace;
	const complex float* pattern;
	const complex float* fftmod_mat;

	long dims_ksp[DIMS];
	long dims_pat[DIMS];

	long strs_ksp[DIMS];
	long strs_pat[DIMS];
};

DEF_TYPEID(data);


static void xupdate_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	const struct data* data = CAST_DOWN(data, _data);

	UNUSED(mu);
	md_zsmul(DIMS, data->dims_ksp, dst, src, 1. / (data->alpha == 0 ? 2. : 3.));
}

static complex float cthresh(float lambda, complex float x)
{
	float norm = cabsf(x);
	float red = norm - lambda;
	return (red > 0.) ? ((red / norm) * x) : 0.;
}



static void robust_consistency(float lambda, const long dims[DIMS], complex float* dst, const complex float* pattern, const complex float* kspace)
{
	assert(1 == dims[MAPS_DIM]);

	size_t size = md_calc_size(DIMS, dims);

	for (unsigned int i = 0; i < size; i++)
		if (1. == pattern[i % (size / dims[COIL_DIM])])
			dst[i] = kspace[i] + cthresh(lambda, dst[i] - kspace[i]);
}



static void sparsity_proj_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	const struct data* data = CAST_DOWN(data, _data);

	const long* dims = data->dims_ksp;

	ifft(DIMS, dims, FFT_FLAGS, dst, src);
	// FIXME fftmod is slow
#if 0
	fftscale(DIMS, dims, FFT_FLAGS, dst, dst);
	ifftmod(DIMS, dims, FFT_FLAGS, dst, dst);
#else
	md_zmulc2(DIMS, dims, data->strs_ksp, dst, data->strs_ksp, dst, data->strs_pat, data->fftmod_mat);
#endif

	operator_p_apply(data->thresh, mu, DIMS, dims, dst, DIMS, dims, dst);

#if 0
	fftmod(DIMS, dims, FFT_FLAGS, dst, dst);
	fftscale(DIMS, dims, FFT_FLAGS, dst, dst);
#else
	md_zmul2(DIMS, dims, data->strs_ksp, dst, data->strs_ksp, dst, data->strs_pat, data->fftmod_mat);
#endif
	fft(DIMS, dims, FFT_FLAGS, dst, dst);
}


static void data_consistency_proj_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);
	const struct data* data = CAST_DOWN(data, _data);

	if (-1. != data->lambda)
		robust_consistency(data->lambda, data->dims_ksp, dst, data->pattern, data->kspace);
	else
		data_consistency(data->dims_ksp, dst, data->pattern, data->kspace, src);
}


static void sense_proj_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	UNUSED(mu);

	const struct data* data = CAST_DOWN(data, _data);

	// assumes normalized sensitivities

	linop_adjoint_unchecked(data->sense_op, data->tmp, src);
	linop_forward_unchecked(data->sense_op, dst, data->tmp);
}


static void proj_del(const operator_data_t* _data)
{
	UNUSED(_data);
}


static float compute_norm(const void* _data, const float* ksp)
{
	const struct data* data = _data;

	float norm = md_znorm(DIMS, data->dims_ksp, (complex float*)ksp);
	//assert(isnormal(norm));
	return norm;
}


void pocs_recon(const long dims[DIMS], const struct operator_p_s* thresh, int maxiter, float alpha, float lambda, complex float* result, const complex float* maps, const complex float* pattern, const complex float* kspace)
{
	struct iter_pocs_conf pconf = iter_pocs_defaults;
	pconf.maxiter = maxiter;

	pocs_recon2(iter2_pocs, &pconf, NULL, dims, thresh, alpha, lambda, result, maps, pattern, kspace);
}

void pocs_recon2(italgo_fun2_t italgo, void* iconf, const struct linop_s* ops[3], const long dims[DIMS], const struct operator_p_s* thresh_op, float alpha, float lambda, complex float* result, const complex float* maps, const complex float* pattern, const complex float* kspace)
{
#ifdef USE_CUDA
	bool use_gpu = cuda_ondevice(kspace);
#else
	bool use_gpu = false;
#endif
	long dims_pat[DIMS];
	long dims_img[DIMS];
	long dims_ksp[DIMS];

	md_select_dims(DIMS, ~(COIL_FLAG | MAPS_FLAG), dims_pat, dims);
	md_select_dims(DIMS, ~(MAPS_FLAG), dims_ksp, dims);
	md_select_dims(DIMS, ~(COIL_FLAG), dims_img, dims);

	long strs_pat[DIMS];
	long strs_ksp[DIMS];

	md_calc_strides(DIMS, strs_pat, dims_pat, CFL_SIZE);
	md_calc_strides(DIMS, strs_ksp, dims_ksp, CFL_SIZE);

	struct data data;
	SET_TYPEID(data, &data);

	data.pattern = pattern;
	data.kspace = kspace;
	data.lambda = lambda;
	data.alpha = alpha;

	md_copy_dims(DIMS, data.dims_ksp, dims_ksp);
	md_copy_dims(DIMS, data.dims_pat, dims_pat);

	md_copy_strides(DIMS, data.strs_ksp, strs_ksp);
	md_copy_strides(DIMS, data.strs_pat, strs_pat);

	data.sense_op = sense_init(dims, FFT_FLAGS|MAPS_FLAG|COIL_FLAG, maps);

	data.thresh = thresh_op;

#ifdef USE_CUDA
	data.tmp = (use_gpu ? md_alloc_gpu : md_alloc)(DIMS, dims_img, CFL_SIZE);
#else
	assert(!use_gpu);
	data.tmp = md_alloc(DIMS, dims_img, CFL_SIZE);
#endif


	complex float* fftmod_mat = md_alloc_sameplace(DIMS, dims_pat, CFL_SIZE, kspace);
	complex float one[1] = { 1. };
	md_fill(DIMS, dims_pat, fftmod_mat, one, CFL_SIZE );
	fftscale(DIMS, dims_pat, FFT_FLAGS, fftmod_mat, fftmod_mat);
	fftmod(DIMS, dims_pat, FFT_FLAGS, fftmod_mat, fftmod_mat);

	data.fftmod_mat = fftmod_mat;


	const struct operator_p_s* sense_proj = operator_p_create(DIMS, dims_ksp, DIMS, dims_ksp, CAST_UP(&data), sense_proj_apply, proj_del);

	const struct operator_p_s* data_consistency_proj = operator_p_create(DIMS, dims_ksp, DIMS, dims_ksp, CAST_UP(&data), data_consistency_proj_apply, proj_del);
	
	const struct operator_p_s* sparsity_proj = NULL;
	if (NULL != thresh_op)
		sparsity_proj = operator_p_create(DIMS, dims_ksp, DIMS, dims_ksp, CAST_UP(&data), sparsity_proj_apply, proj_del);
	else
		sparsity_proj = prox_leastsquares_create(DIMS, dims_ksp, alpha, NULL);

	const struct operator_p_s* prox_ops[3] = { data_consistency_proj, sense_proj, sparsity_proj };
	//const struct operator_p_s* prox_ops[3] = { data_consistency_proj, sense_proj, thresh_op };

	const struct operator_p_s* xupdate_op = operator_p_create(DIMS, dims_ksp, DIMS, dims_ksp, CAST_UP(&data), xupdate_apply, proj_del);

	long size = 2 * md_calc_size(DIMS, dims_ksp);

	md_clear(DIMS, dims_ksp, result, CFL_SIZE);
	italgo(iconf, NULL, (alpha == 0.) ? 2 : 3, prox_ops, ops, NULL, xupdate_op, size, (float*)result, NULL, create_monitor(size, NULL, (void*)&data, compute_norm));

	debug_printf(DP_INFO, "Done\n");

	md_free(data.tmp);
	md_free(fftmod_mat);

	linop_free(data.sense_op);
	operator_p_free(sense_proj);
	operator_p_free(data_consistency_proj);
	operator_p_free(sparsity_proj);
	operator_p_free(xupdate_op);
}


#ifdef USE_CUDA
void pocs_recon_gpu(const long dims[DIMS], const struct operator_p_s* thresh, int maxiter, float alpha, float lambda, complex float* result, const complex float* maps, const complex float* pattern, const complex float* kspace)
{
	struct iter_pocs_conf pconf = iter_pocs_defaults;
	pconf.maxiter = maxiter;

	pocs_recon_gpu2(iter2_pocs, &pconf, NULL, dims, thresh, alpha, lambda, result, maps, pattern, kspace);
}


void pocs_recon_gpu2(italgo_fun2_t italgo, void* iconf, const struct linop_s** ops, const long dims[DIMS], const struct operator_p_s* thresh, float alpha, float lambda, complex float* result, const complex float* maps, const complex float* pattern, const complex float* kspace)
{
	long dims_pat[DIMS];
	long dims_ksp[DIMS];

	md_select_dims(DIMS, ~(COIL_FLAG | MAPS_FLAG), dims_pat, dims);
	md_select_dims(DIMS, ~MAPS_FLAG, dims_ksp, dims);

	complex float* gpu_maps = md_gpu_move(DIMS, dims, maps, CFL_SIZE);
	complex float* gpu_pat = md_gpu_move(DIMS, dims_pat, pattern, CFL_SIZE);
	complex float* gpu_ksp = md_gpu_move(DIMS, dims_ksp, kspace, CFL_SIZE);
	complex float* gpu_result = md_gpu_move(DIMS, dims_ksp, result, CFL_SIZE);

	pocs_recon2(italgo, iconf, ops, dims, thresh, alpha, lambda, gpu_result, gpu_maps, gpu_pat, gpu_ksp);

	md_copy(DIMS, dims_ksp, result, gpu_result, CFL_SIZE);

	md_free(gpu_result);
	md_free(gpu_pat);
	md_free(gpu_ksp);
	md_free(gpu_maps);
}
#endif


