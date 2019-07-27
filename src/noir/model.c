/* Copyright 2013. The Regents of the University of California.
 * Copyright 2017-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-2019 Martin Uecker
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



struct noir_model_conf_s noir_model_conf_defaults = {

	.fft_flags = FFT_FLAGS,
	.cnstcoil_flags = TE_FLAG,
	.ptrn_flags = ~(COIL_FLAG|MAPS_FLAG),
	.rvc = false,
	.noncart = false,
	.a = 220.,
	.b = 32.,
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

	complex float* msk;
	complex float* wghts;
	complex float* ptr;
	complex float* adj_ptr;

	struct noir_model_conf_s conf;
};


DEF_TYPEID(noir_op_s);

static void noir_calc_weights(const struct noir_model_conf_s* conf, const long dims[3], complex float* dst)
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
	PTR_ALLOC(struct noir_op_s, data);
	SET_TYPEID(noir_op_s, data);


	data->conf = *conf;

	md_copy_dims(DIMS, data->dims, dims);

	md_select_dims(DIMS, ~conf->cnstcoil_flags, data->coil_dims, dims);
	md_select_dims(DIMS, ~COIL_FLAG, data->imgs_dims, dims);
	md_select_dims(DIMS, ~MAPS_FLAG, data->data_dims, dims);

	long mask_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, mask_dims, dims);

	long wght_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, wght_dims, dims);

	long ptrn_dims[DIMS];
	md_select_dims(DIMS, conf->ptrn_flags, ptrn_dims, dims);


	data->wghts = md_alloc(DIMS, wght_dims, CFL_SIZE);

	noir_calc_weights(conf, dims, data->wghts);
	fftmod(DIMS, wght_dims, FFT_FLAGS, data->wghts, data->wghts);
	fftscale(DIMS, wght_dims, FFT_FLAGS, data->wghts, data->wghts);

	const struct linop_s* wghts = linop_cdiag_create(DIMS, data->coil_dims, FFT_FLAGS, data->wghts);
	const struct linop_s* wghts_ifft = linop_ifft_create(DIMS, data->coil_dims, FFT_FLAGS);

	data->weights = linop_chain(wghts, wghts_ifft);

	linop_free(wghts);
	linop_free(wghts_ifft);


	const struct linop_s* lop_fft = linop_fft_create(DIMS, data->data_dims, conf->fft_flags);


	data->ptr = md_alloc(DIMS, ptrn_dims, CFL_SIZE);

	md_copy(DIMS, ptrn_dims, data->ptr, psf, CFL_SIZE);
	fftmod(DIMS, ptrn_dims, conf->fft_flags, data->ptr, data->ptr);

	const struct linop_s* lop_pattern = linop_fmac_create(DIMS, data->data_dims, 0, 0, COIL_FLAG, data->ptr);

	const struct linop_s* lop_adj_pattern;

	if (!conf->noncart) {

		lop_adj_pattern = linop_clone(lop_pattern);
		data->adj_ptr = NULL;

	} else {

		data->adj_ptr = md_alloc(DIMS, ptrn_dims, CFL_SIZE);

		md_zfill(DIMS, ptrn_dims, data->adj_ptr, 1.);

		fftmod(DIMS, ptrn_dims, conf->fft_flags, data->adj_ptr, data->adj_ptr);

		lop_adj_pattern = linop_fmac_create(DIMS, data->data_dims, 0, 0, COIL_FLAG, data->adj_ptr);
	}

	data->msk = md_alloc(DIMS, mask_dims, CFL_SIZE);

	if (NULL == mask) {

		md_zfill(DIMS, mask_dims, data->msk, 1.);

	} else {

		md_copy(DIMS, mask_dims, data->msk, mask, CFL_SIZE);
	}

	fftscale(DIMS, mask_dims, FFT_FLAGS, data->msk, data->msk);

	const struct linop_s* lop_mask = linop_cdiag_create(DIMS, data->data_dims, FFT_FLAGS, data->msk);

	const struct linop_s* lop_fft2 = linop_chain(lop_mask, lop_fft);
	linop_free(lop_mask);
	linop_free(lop_fft);

	data->frw = linop_chain(lop_fft2, lop_pattern);
	linop_free(lop_pattern);

	data->adj = linop_chain(lop_fft2, lop_adj_pattern);
	linop_free(lop_fft2);
	linop_free(lop_adj_pattern);



	const struct nlop_s* nlw1 = nlop_tenmul_create(DIMS, data->data_dims, data->imgs_dims, data->coil_dims);

	const struct nlop_s* nlw2 = nlop_from_linop(data->weights);

	data->nl = nlop_chain2(nlw2, 0, nlw1, 1);
	nlop_free(nlw1);
	nlop_free(nlw2);

	const struct nlop_s* frw = nlop_from_linop(data->frw);

	data->nl2 = nlop_chain2(data->nl, 0, frw, 0);

	nlop_free(frw);

	return PTR_PASS(data);
}

static void noir_free(struct noir_op_s* data)
{
	md_free(data->ptr);
	md_free(data->adj_ptr);
	md_free(data->wghts);
	md_free(data->msk);

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


static void noir_fun2(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	void* args2[3] = { args[0], args[1], args[2] };
	nlop_generic_apply_unchecked(data->nl2, 3, args2);
}

static void noir_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	long split = md_calc_size(DIMS, data->imgs_dims);

	noir_fun2(_data, 3, (complex float*[3]){ dst, (complex float*)src, (complex float*)(src + split) });
}

static void noir_derA(const nlop_data_t* _data, complex float* dst, const complex float* img)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	auto der1 = nlop_get_derivative(data->nl, 0, 0);

	linop_forward(der1, DIMS, data->data_dims, dst, DIMS, data->imgs_dims, img);
}

static void noir_derB(const nlop_data_t* _data, complex float* dst, const complex float* coils)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	auto der2 = nlop_get_derivative(data->nl, 0, 1);

	linop_forward(der2, DIMS, data->data_dims, dst, DIMS, data->coil_dims, coils);
}

static void noir_derA2(const nlop_data_t* _data, complex float* dst, const complex float* img)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	complex float* tmp = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, dst);

	noir_derA(_data, tmp, img);
	linop_forward(data->frw, DIMS, data->data_dims, dst, DIMS, data->data_dims, tmp);

	md_free(tmp);
}

static void noir_derB2(const nlop_data_t* _data, complex float* dst, const complex float* coils)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	complex float* tmp = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, dst);

	noir_derB(_data, tmp, coils);
	linop_forward(data->frw, DIMS, data->data_dims, dst, DIMS, data->data_dims, tmp);

	md_free(tmp);
}

static void noir_der2(const nlop_data_t* _data, complex float* dst, const complex float* img, const complex float* coils)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	complex float* tmp = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, img);

	noir_derA(_data, tmp, img);

	complex float* tmp2 = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, img);

	noir_derB(_data, tmp2, coils);

	md_zadd(DIMS, data->data_dims, tmp, tmp, tmp2);

	md_free(tmp2);

	linop_forward(data->frw, DIMS, data->data_dims, dst, DIMS, data->data_dims, tmp);

	md_free(tmp);
}

static void noir_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	long split = md_calc_size(DIMS, data->imgs_dims);

	noir_der2(_data, dst, src, src + split);
}

static void noir_adjA(const nlop_data_t* _data, complex float* img, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	auto der1 = nlop_get_derivative(data->nl, 0, 0);

	linop_adjoint(der1, DIMS, data->imgs_dims, img, DIMS, data->data_dims, src);

	if (data->conf.rvc)
		md_zreal(DIMS, data->imgs_dims, img, img);
}

static void noir_adjB(const nlop_data_t* _data, complex float* coils, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	auto der2 = nlop_get_derivative(data->nl, 0, 1);

	linop_adjoint(der2, DIMS, data->coil_dims, coils, DIMS, data->data_dims, src);
}

static void noir_adjA2(const nlop_data_t* _data, complex float* img, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	complex float* tmp = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, img);

	linop_adjoint(data->adj, DIMS, data->data_dims, tmp, DIMS, data->data_dims, src);
	noir_adjA(_data, img, tmp);

	md_free(tmp);
}

static void noir_adjB2(const nlop_data_t* _data, complex float* coils, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	complex float* tmp = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, coils);

	linop_adjoint(data->adj, DIMS, data->data_dims, tmp, DIMS, data->data_dims, src);
	noir_adjB(_data, coils, tmp);

	md_free(tmp);
}

static void noir_adj2(const nlop_data_t* _data, complex float* img, complex float* coils, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	complex float* tmp = md_alloc_sameplace(DIMS, data->data_dims, CFL_SIZE, img);

	linop_adjoint(data->adj, DIMS, data->data_dims, tmp, DIMS, data->data_dims, src);

	noir_adjB(_data, coils, tmp);
	noir_adjA(_data, img, tmp);

	md_free(tmp);
}

static void noir_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(noir_op_s, _data);

	long split = md_calc_size(DIMS, data->imgs_dims);

	noir_adj2(_data, dst, dst + split, src);
}

struct noir_s noir_create3(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
	struct noir_op_s* data = noir_init(dims, mask, psf, conf);

	long idims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, idims, dims);

	long odims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, odims, dims);

	long cdims[DIMS];
	md_select_dims(DIMS, ~conf->cnstcoil_flags, cdims, dims);


	long nl_odims[1][DIMS];
	md_copy_dims(DIMS, nl_odims[0], odims);

	long nl_ostr[1][DIMS];
	md_calc_strides(DIMS, nl_ostr[0], odims, CFL_SIZE);

	long nl_idims[2][DIMS];
	md_copy_dims(DIMS, nl_idims[0], idims);
	md_copy_dims(DIMS, nl_idims[1], cdims);

	long nl_istr[2][DIMS];
	md_calc_strides(DIMS, nl_istr[0], idims, CFL_SIZE);
	md_calc_strides(DIMS, nl_istr[1], cdims, CFL_SIZE);

	struct noir_s ret = { .linop = data->weights, .noir_op = data };

	ret.nlop = nlop_generic_create2(1, DIMS, nl_odims, nl_ostr, 2, DIMS, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)),
			noir_fun2, (nlop_fun_t[2][1]){ { noir_derA2 }, { noir_derB2 } },
			(nlop_fun_t[2][1]){ { noir_adjA2 }, { noir_adjB2 } }, NULL, NULL, noir_del);

	return ret;
}


struct noir_s noir_create2(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
	assert(!conf->noncart);
	assert(!conf->rvc);

	struct noir_op_s* data = noir_init(dims, mask, psf, conf);
	struct nlop_s* nlop = data->nl2;
	return (struct noir_s){ .nlop = nlop, .linop = data->weights, .noir_op = data };
}

struct noir_s noir_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
#if 1
	struct noir_op_s* data = noir_init(dims, mask, psf, conf);

	long idims[DIMS];
	md_select_dims(DIMS, ~(COIL_FLAG|conf->cnstcoil_flags), idims, dims);

	long edims[DIMS];
	md_select_dims(DIMS, conf->cnstcoil_flags, edims, dims);

	idims[COIL_DIM] = dims[COIL_DIM] + md_calc_size(DIMS, edims); // add images

	long odims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, odims, dims);

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
	float v22 = md_zscalar_real2(D, dims, v2_strs, v2, v2_strs, v2); // since it is real anyway

	complex float v12 = md_zscalar2(D, dims, v1_strs, v1, v2_strs, v2) / v22;

	if (!isfinite(crealf(v12)) || !isfinite(cimagf(v12)))
		v12 = 0.;

	md_zaxpy2(D, dims, ostrs, optr, v12, v2_strs, v2);
}





// FIXME: review dimensions
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

