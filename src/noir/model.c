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

#include "noir/utils.h"

#include "model.h"



struct noir_model_conf_s noir_model_conf_defaults = {

	.fft_flags = FFT_FLAGS,
	.cnstcoil_flags = 0u,
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

	const struct nlop_s* nl;
	/*const*/ struct nlop_s* nl2;

	struct noir_model_conf_s conf;
};


DEF_TYPEID(noir_op_s);

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


	complex float* wghts = md_alloc(DIMS, wght_dims, CFL_SIZE);

	noir_calc_weights(conf->a, conf->b, dims, wghts);
	fftmod(DIMS, wght_dims, FFT_FLAGS, wghts, wghts);
	fftscale(DIMS, wght_dims, FFT_FLAGS, wghts, wghts);

	const struct linop_s* lop_wghts = linop_cdiag_create(DIMS, data->coil_dims, FFT_FLAGS, wghts);
	const struct linop_s* lop_wghts_ifft = linop_ifft_create(DIMS, data->coil_dims, FFT_FLAGS);

	md_free(wghts);

	data->weights = linop_chain_FF(lop_wghts, lop_wghts_ifft);



	const struct linop_s* lop_fft = linop_fft_create(DIMS, data->data_dims, conf->fft_flags);


	complex float* ptr = md_alloc(DIMS, ptrn_dims, CFL_SIZE);

	md_copy(DIMS, ptrn_dims, ptr, psf, CFL_SIZE);
	fftmod(DIMS, ptrn_dims, conf->fft_flags, ptr, ptr);

	const struct linop_s* lop_pattern = linop_fmac_create(DIMS, data->data_dims, 0, 0, ~conf->ptrn_flags, ptr);
	md_free(ptr);

	if (conf->noncart) {

		complex float* adj_ptr = md_alloc(DIMS, ptrn_dims, CFL_SIZE);

		md_zfill(DIMS, ptrn_dims, adj_ptr, 1.);

		fftmod(DIMS, ptrn_dims, conf->fft_flags, adj_ptr, adj_ptr);

		const struct linop_s* lop_adj_pattern = linop_fmac_create(DIMS, data->data_dims, 0, 0, ~conf->ptrn_flags, adj_ptr);

		md_free(adj_ptr);

		const struct linop_s* lop_tmp = linop_from_ops(lop_pattern->forward, lop_adj_pattern->adjoint, NULL, NULL);

		linop_free(lop_adj_pattern);
		linop_free(lop_pattern);

		lop_pattern = lop_tmp;
	}

	complex float* msk = md_alloc(DIMS, mask_dims, CFL_SIZE);

	if (NULL == mask) {

		md_zfill(DIMS, mask_dims, msk, 1.);

	} else {

		md_copy(DIMS, mask_dims, msk, mask, CFL_SIZE);
	}

	fftscale(DIMS, mask_dims, FFT_FLAGS, msk, msk);

	const struct linop_s* lop_mask = linop_cdiag_create(DIMS, data->imgs_dims, FFT_FLAGS, msk);
	md_free(msk);

	data->frw = linop_chain_FF(lop_fft, lop_pattern);


	const struct nlop_s* nlw1 = nlop_tenmul_create(DIMS, data->data_dims, data->imgs_dims, data->coil_dims);
	nlw1 = nlop_chain2_swap_FF(nlop_from_linop_F(lop_mask), 0, nlw1, 0);

	const struct nlop_s* nlw2 = nlop_from_linop(data->weights);
	data->nl = nlop_chain2_FF(nlw2, 0, nlw1, 1);

	if (conf->rvc) {

		const struct nlop_s* nlop_zreal = nlop_from_linop_F(linop_zreal_create(DIMS, data->imgs_dims));
		data->nl = nlop_chain2_swap_FF(nlop_zreal, 0, data->nl, 0);
	}

	const struct nlop_s* frw = nlop_from_linop(data->frw);
	data->nl2 = nlop_chain2(data->nl, 0, frw, 0);

	nlop_free(frw);

	return PTR_PASS(data);
}

static void noir_free(struct noir_op_s* data)
{
	linop_free(data->frw);
	linop_free(data->weights);

	nlop_free(data->nl);
	nlop_free(data->nl2);

	xfree(data);
}

static void noir_del(const void* _data)
{
	noir_free(CAST_DOWN(noir_op_s, (const nlop_data_t*)_data));
}

void noir_forw_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void noir_back_coils(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}




struct noir_s noir_create3(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
	return noir_create2(dims, mask, psf, conf);
}


struct noir_s noir_create2(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
	struct noir_op_s* data = noir_init(dims, mask, psf, conf);
	struct nlop_s* nlop = (struct nlop_s*)nlop_attach(data->nl2, data, noir_del);
	return (struct noir_s){ .nlop = nlop, .linop = data->weights, .noir_op = data };
}

struct noir_s noir_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{
	struct noir_s ret = noir_create2(dims, mask, psf, conf);
	ret.nlop = nlop_flatten_F(ret.nlop);
	return ret;
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
	md_select_dims(DIMS, ~MAPS_FLAG, single_map_dims, data->coil_dims);

	long single_map_strs[DIMS];
	md_calc_strides(DIMS, single_map_strs, single_map_dims, CFL_SIZE);

	long data_strs[DIMS];
	md_calc_strides(DIMS, data_strs, data->coil_dims, CFL_SIZE);

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

