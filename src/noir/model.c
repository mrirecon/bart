/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2018 Martin Uecker
 *
 *
 * Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized nonlinear
 * inversion – Joint estimation of coil sensitivities and image content. 
 * Magn Reson Med 2008; 60:674-682.
 */


#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "nlops/nlop.h"
#include "nlops/tenmul.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "num/fft.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "model.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif



struct noir_model_conf_s noir_model_conf_defaults = {

	.fft_flags = FFT_FLAGS,
	.rvc = false,
	.use_gpu = false,
	.noncart = false,
	.a = 220.,
	.b = 32.,
	.pattern_for_each_coil = false,
};



struct noir_op_s {

	INTERFACE(nlop_data_t);

	long dims[DIMS];

	long data_dims[DIMS];
	long coil_dims[DIMS];
	long imgs_dims[DIMS];

	const struct linop_s* weights;
	const struct linop_s* frw;
	const struct linop_s* adj;

	const struct nlop_s* nl;
	/*const*/ struct nlop_s* nl2;

	complex float* weights_array;
	complex float* pattern_array;
	complex float* adj_pattern_array;
	complex float* mask_array;
	complex float* tmp;

	struct noir_model_conf_s conf;
};


DEF_TYPEID(noir_op_s);

static void noir_calc_weights(const struct noir_model_conf_s *conf, const long dims[3], complex float* dst)
{
	unsigned int flags = 0;

	for (int i = 0; i < 3; i++)
		if (1 != dims[i])
			flags = MD_SET(flags, i);

	klaplace(3, dims, flags, dst);
	md_zsmul(3, dims, dst, dst, conf->a);
	md_zsadd(3, dims, dst, dst, 1.);
	md_zspow(3, dims, dst, dst, -conf->b / 2.);	// 1 + 220. \Laplace^16
}


static struct noir_op_s* noir_init(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = conf->use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!conf->use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct noir_op_s, data);
	SET_TYPEID(noir_op_s, data);


	data->conf = *conf;

	md_copy_dims(DIMS, data->dims, dims);

	md_select_dims(DIMS, conf->fft_flags|COIL_FLAG|MAPS_FLAG, data->coil_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|MAPS_FLAG, data->imgs_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COIL_FLAG, data->data_dims, dims);

	long mask_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, mask_dims, dims);

	long wght_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, wght_dims, dims);

	long ptrn_dims[DIMS];
	unsigned int ptrn_flags;
	if (!conf->pattern_for_each_coil) {

		md_select_dims(DIMS, conf->fft_flags, ptrn_dims, dims);
		ptrn_flags = ~(conf->fft_flags);
	} else {

		md_select_dims(DIMS, conf->fft_flags|COIL_FLAG, ptrn_dims, dims);
		ptrn_flags = ~(conf->fft_flags|COIL_FLAG);
	}



	data->weights_array = md_alloc(DIMS, wght_dims, CFL_SIZE);

	noir_calc_weights(conf, dims, data->weights_array);
	fftmod(DIMS, wght_dims, FFT_FLAGS, data->weights_array, data->weights_array);
	fftscale(DIMS, wght_dims, FFT_FLAGS, data->weights_array, data->weights_array);

	const struct linop_s* tmp_weights = linop_cdiag_create(DIMS, data->coil_dims, FFT_FLAGS, data->weights_array);
	const struct linop_s* tmp_ifft = linop_ifft_create(DIMS, data->coil_dims, FFT_FLAGS);
	data->weights = linop_chain(tmp_weights, tmp_ifft);
	linop_free(tmp_weights);
	linop_free(tmp_ifft);



	const struct linop_s* lop_fft = linop_fft_create(DIMS, data->data_dims, conf->fft_flags);


	data->pattern_array = md_alloc(DIMS, ptrn_dims, CFL_SIZE);

	md_copy(DIMS, ptrn_dims, data->pattern_array, psf, CFL_SIZE);
	fftmod(DIMS, ptrn_dims, conf->fft_flags, data->pattern_array, data->pattern_array);


	const struct linop_s* lop_pattern = linop_fmac_create(DIMS, data->data_dims, ~(conf->fft_flags|COIL_FLAG), ~(conf->fft_flags|COIL_FLAG), ptrn_flags, data->pattern_array);

	const struct linop_s* lop_adj_pattern;

	if (!conf->noncart) {

		lop_adj_pattern = linop_clone(lop_pattern);
		data->adj_pattern_array = NULL;

	} else {

		data->adj_pattern_array = md_alloc(DIMS, ptrn_dims, CFL_SIZE);
		md_zfill(DIMS, ptrn_dims, data->adj_pattern_array, 1.);
		fftmod(DIMS, ptrn_dims, conf->fft_flags, data->adj_pattern_array, data->adj_pattern_array);

		lop_adj_pattern = linop_fmac_create(DIMS, data->data_dims, ~(conf->fft_flags|COIL_FLAG), ~(conf->fft_flags|COIL_FLAG), ptrn_flags, data->adj_pattern_array);
	}



	data->mask_array = md_alloc(DIMS, mask_dims, CFL_SIZE);

	if (NULL == mask) {

		assert(!conf->use_gpu);
		md_zfill(DIMS, mask_dims, data->mask_array, 1.);

	} else {

		md_copy(DIMS, mask_dims, data->mask_array, mask, CFL_SIZE);
	}

	fftscale(DIMS, mask_dims, FFT_FLAGS, data->mask_array, data->mask_array);

	const struct linop_s* lop_mask = linop_cdiag_create(DIMS, data->data_dims, FFT_FLAGS, data->mask_array);

	const struct linop_s* tmp_lop_fft = linop_chain(lop_mask, lop_fft);
	linop_free(lop_fft);
	linop_free(lop_mask);

	data->frw = linop_chain(tmp_lop_fft, lop_pattern);
	data->adj = linop_chain(tmp_lop_fft, lop_adj_pattern);
	linop_free(tmp_lop_fft);
	linop_free(lop_pattern);
	linop_free(lop_adj_pattern);

	data->tmp = my_alloc(DIMS, data->data_dims, CFL_SIZE);


	const struct nlop_s* tmp_tenmul = nlop_tenmul_create(DIMS, data->data_dims, data->imgs_dims, data->coil_dims);

	const struct nlop_s* nlw = nlop_from_linop(data->weights);

	data->nl = nlop_chain2(nlw, 0, tmp_tenmul, 1);

	nlop_free(tmp_tenmul);
	nlop_free(nlw);

	const struct nlop_s* frw = nlop_from_linop(data->frw);

	data->nl2 = nlop_chain2(data->nl, 0, frw, 0);

	nlop_free(frw);

	return PTR_PASS(data);
}

static void noir_free(struct noir_op_s* data)
{
	md_free(data->weights_array);
	md_free(data->pattern_array);
	md_free(data->adj_pattern_array);
	md_free(data->mask_array);
	md_free(data->tmp);

	linop_free(data->frw);
	linop_free(data->adj);
	linop_free(data->weights);

	nlop_free(data->nl);
	nlop_free(data->nl2);

	xfree(data);
}

static void noir_del(const nlop_data_t* _data)
{
	noir_free(CAST_DOWN(noir_op_s, _data));
}

void noir_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void noir_back_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}


static void noir_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{	
	const auto data = CAST_DOWN(noir_op_s, _data);

	long split = md_calc_size(DIMS, data->imgs_dims);

	nlop_generic_apply_unchecked(data->nl2, 3, (void*[3]){ dst, (void*)(src), (void*)(src + split) });
}


static void noir_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	long split = md_calc_size(DIMS, data->imgs_dims);
#if 1
	auto der1 = nlop_get_derivative(data->nl, 0, 0);
	auto der2 = nlop_get_derivative(data->nl, 0, 1);

	linop_forward(der1, DIMS, data->data_dims, data->tmp, DIMS, data->imgs_dims, src);

	complex float* tmp2 = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, src);

	linop_forward(der2, DIMS, data->data_dims, tmp2, DIMS, data->coil_dims, src + split);
	md_zadd(DIMS, data->data_dims, data->tmp, data->tmp, tmp2);
	md_free(tmp2);

	linop_forward(data->frw, DIMS, data->data_dims, dst, DIMS, data->data_dims, data->tmp);
#else
	auto der1 = nlop_get_derivative(data->nl2, 0, 0);
	auto der2 = nlop_get_derivative(data->nl2, 0, 1);

	linop_forward(der1, DIMS, data->data_dims, dst, DIMS, data->imgs_dims, src);

	complex float* tmp2 = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, src);

	linop_forward(der2, DIMS, data->data_dims, tmp2, DIMS, data->coil_dims, src + split);
	md_zadd(DIMS, data->data_dims, dst, dst, tmp2);
	md_free(tmp2);
#endif
}


static void noir_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	long split = md_calc_size(DIMS, data->imgs_dims);

	auto der1 = nlop_get_derivative(data->nl, 0, 0);
	auto der2 = nlop_get_derivative(data->nl, 0, 1);

	linop_adjoint(data->adj, DIMS, data->data_dims, data->tmp, DIMS, data->data_dims, src);

	linop_adjoint(der2, DIMS, data->coil_dims, dst + split, DIMS, data->data_dims, data->tmp);

	linop_adjoint(der1, DIMS, data->imgs_dims, dst, DIMS, data->data_dims, data->tmp);

	if (data->conf.rvc)
		md_zreal(DIMS, data->imgs_dims, dst, dst);
}



struct noir_s noir_create2(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
	assert(!conf->noncart);
	assert(!conf->rvc);

	struct noir_op_s* data = noir_init(dims, mask, psf, conf);
	struct nlop_s* nlop = data->nl2;
//	noir_free(data);
	return (struct noir_s){ .nlop = nlop, .linop = data->weights };
}

struct noir_s noir_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
#if 1
	struct noir_op_s* data = noir_init(dims, mask, psf, conf);

	long idims[DIMS];
	md_select_dims(DIMS, conf->fft_flags|MAPS_FLAG|CSHIFT_FLAG, idims, dims);
	idims[COIL_DIM] = dims[COIL_DIM] + 1; // add image

	long odims[DIMS];
	md_select_dims(DIMS, conf->fft_flags|COIL_FLAG|CSHIFT_FLAG, odims, dims);

	struct noir_s ret = { .linop = data->weights, .noir_op = data };
	ret.nlop = nlop_create(DIMS, odims, DIMS, idims, CAST_UP(PTR_PASS(data)), noir_fun, noir_der, noir_adj, NULL, NULL, noir_del);

	return ret;
#else
	// less efficient than the manuel coded functions

	struct noir_s ret = noir_create2(dims, mask, psf, conf);
	ret.nlop = nlop_flatten(ret.nlop);
	return ret;
#endif
}



__attribute__((optimize("-fno-finite-math-only")))
static void proj_add(unsigned int D, const long dims[D], const long ostrs[D],
			complex float* optr, const long v1_strs[D], complex float* v1, const long v2_strs[D], complex float* v2)
{
#ifdef USE_CUDA
	if (cuda_ondevice(v1))
		error("md_zscalar is far too slow on the GPU, refusing to run...\n");
#endif
	float v22 = md_zscalar_real2(D, dims, v2_strs, v2, v2_strs, v2); // since it is real anyway

	complex float v12 = md_zscalar2(D, dims, v1_strs, v1, v2_strs, v2) / v22;

	if (!isfinite(crealf(v12)) || !isfinite(cimagf(v12)))
		v12 = 0.;

	md_zaxpy2(D, dims, ostrs, optr, v12, v2_strs, v2);
}





void noir_orthogonalize(struct noir_s* op, complex float* coils)
{
	struct noir_op_s* data = op->noir_op;

	// orthogonalization of the coil profiles
	long nmaps = data->imgs_dims[MAPS_DIM];

	if (1L == nmaps)
		return;

	long single_map_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, single_map_dims, data->dims);

	long single_map_strs[DIMS];
	md_calc_strides(DIMS, single_map_strs, single_map_dims, CFL_SIZE);

	long data_strs[DIMS];
	md_calc_strides(DIMS, data_strs, data->dims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(DIMS, single_map_dims, CFL_SIZE, coils);

	for (long map = 0L; map < nmaps; ++map) {

		complex float* map_ptr = (void*)coils + map * data_strs[MAPS_DIM];

		md_clear(DIMS, single_map_dims, tmp, CFL_SIZE);

		for (long prev = 0L; prev < map; ++prev) {

			complex float* prev_map_ptr = (void*)coils + prev * data_strs[MAPS_DIM];

			proj_add(DIMS, single_map_dims, single_map_strs, tmp, single_map_strs, map_ptr, data_strs, prev_map_ptr);
		}

		md_zsub2(DIMS, single_map_dims, data_strs, map_ptr, data_strs, map_ptr, single_map_strs, tmp);
	}

	md_free(tmp);
}

