/* Copyright 2021-2022. Uecker Lab. University Center Göttingen.
 * Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/types.h"
#include "misc/shrdptr.h"

#include "noncart/nufft.h"

#include "num/multiplace.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/rand.h"
#include "num/fft.h"
#include "num/init.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "iter/misc.h"
#include "iter/iter.h"
#include "iter/iter2.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/norm_inv.h"

#include "mri_ops.h"

static bool multigpu = false;

void mri_ops_activate_multigpu(void)
{
	multigpu = true;
}

void mri_ops_deactivate_multigpu(void)
{
	multigpu = false;
}

struct config_nlop_mri_s {

	int N;
	int ND;

	long* img_dims;
	long* col_dims;
	long* cim_dims;
	long* ksp_dims;
	long* bas_dims;
	long* pat_dims;
	long* trj_dims;
	long* psf_dims;

	long* pat_dims_merged;
	long* fftmod_precomp_img_dims;
	long* fftmod_precomp_ksp_dims;
	struct multiplace_array_s* fftmod_precomp_img;
	struct multiplace_array_s* fftmod_precomp_ksp;

	const complex float* basis;

	_Bool noncart;
	struct nufft_conf_s nufft_conf;

	struct shared_obj_s sptr;
};

static void sense_config_del(const struct shared_obj_s* sptr)
{
	const struct config_nlop_mri_s* x = CONTAINER_OF(sptr, const struct config_nlop_mri_s, sptr);

	xfree(x->img_dims);
	xfree(x->col_dims);
	xfree(x->cim_dims);
	xfree(x->ksp_dims);
	xfree(x->bas_dims);
	xfree(x->pat_dims);
	xfree(x->trj_dims);
	xfree(x->psf_dims);

	if (x->fftmod_precomp_img != x->fftmod_precomp_ksp)
		multiplace_free(x->fftmod_precomp_img);

	multiplace_free(x->fftmod_precomp_ksp);

	xfree(x->pat_dims_merged);
	xfree(x->fftmod_precomp_img_dims);
	xfree(x->fftmod_precomp_ksp_dims);

	//only save reference
	//md_free(x->basis);

	xfree(x);
}

void sense_model_config_free(const struct config_nlop_mri_s* x)
{
	if (NULL == x)
		return;

	shared_obj_destroy(&x->sptr);
}

static const struct config_nlop_mri_s* sense_model_config_ref(const struct config_nlop_mri_s* x)
{
	if (NULL != x)
		shared_obj_ref(&x->sptr);

	return x;
}

static struct config_nlop_mri_s* sense_model_config_init(int N, int ND)
{
	PTR_ALLOC(struct config_nlop_mri_s, result);

	*result = (struct config_nlop_mri_s) {

		.N = N,
		.ND = ND,

		.img_dims = *TYPE_ALLOC(long[N]),
		.col_dims = *TYPE_ALLOC(long[N]),
		.cim_dims = *TYPE_ALLOC(long[N]),
		.ksp_dims = *TYPE_ALLOC(long[N]),
		.bas_dims = *TYPE_ALLOC(long[N]),
		.pat_dims = *TYPE_ALLOC(long[N]),
		.trj_dims = *TYPE_ALLOC(long[N]),
		.psf_dims = *TYPE_ALLOC(long[ND]),

		.pat_dims_merged = *TYPE_ALLOC(long[N]),
		.fftmod_precomp_img_dims = *TYPE_ALLOC(long[N]),
		.fftmod_precomp_ksp_dims = *TYPE_ALLOC(long[N]),

		.fftmod_precomp_img = NULL,
		.fftmod_precomp_ksp = NULL,

		.basis = NULL,
	};

	shared_obj_init(&(result->sptr), sense_config_del);

	md_singleton_dims(N, result->img_dims);
	md_singleton_dims(N, result->col_dims);
	md_singleton_dims(N, result->cim_dims);
	md_singleton_dims(N, result->ksp_dims);
	md_singleton_dims(N, result->bas_dims);
	md_singleton_dims(N, result->pat_dims);
	md_singleton_dims(N, result->trj_dims);
	md_singleton_dims(ND, result->psf_dims);

	return PTR_PASS(result);
}

struct config_nlop_mri_s* sense_model_config_cart_create(int N, const long ksp_dims[N], const long img_dims[N], const long col_dims[N], const long pat_dims[N])
{
	assert(N == DIMS);
	struct config_nlop_mri_s* result = sense_model_config_init(N, N);

	result->noncart = false;

	md_copy_dims(N, result->ksp_dims, ksp_dims);
	md_copy_dims(N, result->img_dims, img_dims);
	md_copy_dims(N, result->pat_dims, pat_dims);
	md_copy_dims(N, result->psf_dims, pat_dims);
	md_copy_dims(N, result->col_dims, col_dims);

	long ksp_dims2[N];
	md_copy_dims(N, ksp_dims2, ksp_dims);
	md_copy_dims(3, ksp_dims2, col_dims);

	long max_dims[N];
	md_singleton_dims(N, max_dims);

	md_max_dims(N, ~0UL, max_dims, ksp_dims2, max_dims);
	md_max_dims(N, ~0UL, max_dims, img_dims, max_dims);
	md_max_dims(N, ~0UL, max_dims, col_dims, max_dims);

	md_select_dims(N, ~MAPS_FLAG, result->cim_dims, max_dims);

	md_select_dims(N, FFT_FLAGS, result->fftmod_precomp_ksp_dims, result->ksp_dims);

	complex float* tmp = md_alloc(N, result->fftmod_precomp_ksp_dims, CFL_SIZE);

	md_zfill(N, result->fftmod_precomp_ksp_dims, tmp, 1. / sqrt(sqrt(md_calc_size(N, result->fftmod_precomp_ksp_dims))));

	fftmod(N, result->fftmod_precomp_ksp_dims, FFT_FLAGS, tmp, tmp);

	result->fftmod_precomp_ksp = multiplace_move(N, result->fftmod_precomp_ksp_dims, CFL_SIZE, tmp);

	md_free(tmp);


	md_select_dims(N, md_nontriv_dims(N, result->fftmod_precomp_ksp_dims) | md_nontriv_dims(N, result->pat_dims),result->pat_dims_merged, result->ksp_dims);

	if (!md_check_equal_dims(N, ksp_dims, ksp_dims2, ~0UL)) {

		md_select_dims(N, FFT_FLAGS, result->fftmod_precomp_img_dims, result->img_dims);

		complex float* tmp = md_alloc(N, result->fftmod_precomp_img_dims, CFL_SIZE);

		md_resize_center(N, result->fftmod_precomp_img_dims, tmp, result->fftmod_precomp_ksp_dims, multiplace_read(result->fftmod_precomp_ksp, tmp), CFL_SIZE);

		result->fftmod_precomp_img = multiplace_move(N, result->fftmod_precomp_img_dims, CFL_SIZE, tmp);

		md_free(tmp);

	} else {

		md_copy_dims(N, result->fftmod_precomp_img_dims, result->fftmod_precomp_ksp_dims);

		result->fftmod_precomp_img = result->fftmod_precomp_ksp;

	}

	return result;
}


struct config_nlop_mri_s* sense_model_config_noncart_create(int N,
	const long trj_dims[N], const long wgh_dims[N], const long ksp_dims[N],
	const long cim_dims[N],	const long img_dims[N], const long col_dims[N],
	const long bas_dims[N], const complex float* basis,
	struct nufft_conf_s conf)
{
	assert(N == DIMS);
	struct config_nlop_mri_s* result = sense_model_config_init(N, N + 1);

	md_copy_dims(N, result->ksp_dims, ksp_dims);
	md_copy_dims(N, result->img_dims, img_dims);
	md_copy_dims(N, result->pat_dims, wgh_dims);
	md_copy_dims(N, result->col_dims, col_dims);
	md_copy_dims(N, result->bas_dims, bas_dims);
	md_copy_dims(N, result->trj_dims, trj_dims);
	md_copy_dims(N, result->cim_dims, cim_dims);

	long max_dims[N];
	md_singleton_dims(N, max_dims);

	md_max_dims(N, ~0UL, max_dims, cim_dims, max_dims);
	md_max_dims(N, ~0UL, max_dims, img_dims, max_dims);
	md_max_dims(N, ~0UL, max_dims, col_dims, max_dims);

	md_select_dims(N, 7, result->psf_dims, max_dims);
	result->psf_dims[N] = 1;
	for (int i = 0; i < 3; i++)
		if (1 != result->psf_dims[i])
			result->psf_dims[N] *= 2;

	for (int i = 3; i < N; i++)
		result->psf_dims[i] = MAX(result->trj_dims[i], result->pat_dims[i]);

	if (NULL != basis) {

		result->psf_dims[6] = result->bas_dims[6];
		result->psf_dims[5] = result->bas_dims[6];
	}

	result->basis = basis;
	result->nufft_conf = conf;
	result->noncart = true;

	return result;
}





struct sense_model_s {

	const struct config_nlop_mri_s* config;

	const struct linop_s* sense;
	const struct linop_s* coils;
	const struct linop_s* pattern;
	const struct linop_s* nufft;

	const struct linop_s* nufft_loss;

	struct shared_obj_s sptr;
};

static void sense_model_del(const struct shared_obj_s* sptr)
{
	const struct sense_model_s* x = CONTAINER_OF(sptr, const struct sense_model_s, sptr);

	linop_free(x->sense);

	linop_free(x->coils);
	linop_free(x->pattern);
	linop_free(x->nufft);
	linop_free(x->nufft_loss);

	sense_model_config_free(x->config);

	xfree(x);
}

void sense_model_free(const struct sense_model_s* x)
{
	if (NULL == x)
		return;

	shared_obj_destroy(&x->sptr);
}

static const struct sense_model_s* sense_model_ref(const struct sense_model_s* x)
{
	if (NULL != x)
		shared_obj_ref(&x->sptr);

	return x;
}

static struct sense_model_s* mri_sense_init(void)
{
	PTR_ALLOC(struct sense_model_s, result);

	*result = (struct sense_model_s) {

		.config = NULL,

		.sense = NULL,
		.coils = NULL,
		.pattern = NULL,
		.nufft = NULL,
		.nufft_loss = NULL,
	};

	shared_obj_init(&(result->sptr), sense_model_del);

	return PTR_PASS(result);
}

struct sense_model_s* sense_model_create(const struct config_nlop_mri_s* config)
{
	struct sense_model_s* result = mri_sense_init();

	result->config = sense_model_config_ref(config);

	if (config->noncart) {

		result->coils = linop_fmac_dims_create(config->N, config->cim_dims, config->img_dims, config->col_dims, NULL);

		result->nufft = nufft_create2(DIMS, config->ksp_dims, config->cim_dims, config->trj_dims, NULL, config->pat_dims, NULL, config->bas_dims, config->basis, config->nufft_conf);

		result->sense = linop_chain(result->coils, result->nufft);

	} else {

		debug_print_dims(DP_INFO, config->N, config->cim_dims);
		debug_print_dims(DP_INFO, config->N, config->col_dims);
		debug_print_dims(DP_INFO, config->N, config->img_dims);

		result->coils = linop_fmac_dims_create(config->N, config->cim_dims, config->img_dims, config->col_dims, NULL);

		result->pattern = linop_cdiag_create(config->N, config->ksp_dims, md_nontriv_dims(config->N, config->pat_dims) | md_nontriv_dims(config->N, config->fftmod_precomp_ksp_dims), NULL);

		result->sense = linop_clone(result->coils);

		if (!md_check_equal_dims(config->N, config->cim_dims, config->ksp_dims, ~0UL))
			result->sense = linop_chain_FF(result->sense, linop_resize_center_create(config->N, config->ksp_dims, config->cim_dims));

		result->sense = linop_chain_FF(result->sense, linop_fft_create(config->N, config->ksp_dims, FFT_FLAGS));
		result->sense = linop_chain_FF(result->sense, linop_clone(result->pattern));
	}

	return result;
}


struct sense_model_s* sense_model_normal_create(const struct config_nlop_mri_s* config)
{
	assert(NULL != config);
	struct sense_model_s* result = mri_sense_init();

	result->config = sense_model_config_ref(config);

	if (config->noncart) {

		result->coils = linop_fmac_dims_create(config->N, config->cim_dims, config->img_dims, config->col_dims, NULL);

		result->nufft = nufft_create_normal(config->N, config->cim_dims, config->ND, config->psf_dims, NULL, NULL != config->basis, config->nufft_conf);

		nufft_get_psf_dims(result->nufft, config->ND, config->psf_dims);

		result->sense = linop_chain(result->coils, result->nufft);

	} else {

		result->coils = linop_fmac_dims_create(config->N, config->cim_dims, config->img_dims, config->col_dims, NULL);
		result->pattern = linop_cdiag_create(config->N, config->cim_dims, md_nontriv_dims(config->N, config->pat_dims) | md_nontriv_dims(config->N, config->fftmod_precomp_ksp_dims), NULL);

		result->sense = linop_chain_FF(linop_clone(result->coils), linop_fft_create(config->N, config->cim_dims, FFT_FLAGS));
		result->sense = linop_chain_FF(result->sense, linop_clone(result->pattern));
	}

	return result;
}


int sense_model_get_N(struct config_nlop_mri_s* model)
{
	return model->N;
}

void sense_model_get_img_dims(struct config_nlop_mri_s* model, int N, long img_dims[N])
{
	assert(N == model->N);
	md_copy_dims(N, img_dims, model->img_dims);
}

void sense_model_get_col_dims(struct config_nlop_mri_s* model, int N, long col_dims[N])
{
	assert(N == model->N);
	md_copy_dims(N, col_dims, model->col_dims);
}

void sense_model_get_cim_dims(struct config_nlop_mri_s* model, int N, long cim_dims[N])
{
	assert(N == model->N);
	md_copy_dims(N, cim_dims, model->cim_dims);
}

void sense_model_get_ksp_dims(struct config_nlop_mri_s* model, int N, long ksp_dims[N])
{
	assert(N == model->N);
	md_copy_dims(N, ksp_dims, model->ksp_dims);
}

bool sense_model_get_noncart(const struct config_nlop_mri_s* model)
{
	assert(NULL != model);
	return model->noncart;
}



struct sense_model_set_data_s {

	nlop_data_t super;

	bool output_psf;

	int N;
	long* dims;

	const struct sense_model_s* model;
};

DEF_TYPEID(sense_model_set_data_s);

static void sense_model_set_data_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(sense_model_set_data_s, _data);

	int i = 0;
	complex float* dst = args[i++];
	complex float* dst_psf = d->output_psf ? args[i++] : NULL;

	const complex float* src = args[i++];
	const complex float* coil = args[i++];
	const complex float* pattern = args[i++];

	md_copy(d->N, d->dims, dst, src, CFL_SIZE);

	if (NULL != dst_psf)
		md_copy(d->model->config->ND, d->model->config->psf_dims, dst_psf, pattern, CFL_SIZE);

	if (NULL != d->model->pattern) {

		auto m = d->model;

		complex float* tmp_coil = md_alloc_sameplace(m->config->N, m->config->col_dims, CFL_SIZE, coil);
		complex float* tmp_pattern = md_alloc_sameplace(m->config->N, m->config->pat_dims_merged, CFL_SIZE, coil);

		md_ztenmul(m->config->N, m->config->col_dims, tmp_coil, m->config->col_dims, coil, m->config->fftmod_precomp_img_dims, multiplace_read(m->config->fftmod_precomp_img, coil));
		md_ztenmul(m->config->N, m->config->pat_dims_merged, tmp_pattern, m->config->pat_dims, pattern, m->config->fftmod_precomp_ksp_dims, multiplace_read(m->config->fftmod_precomp_ksp, pattern));

		linop_fmac_set_tensor(m->coils, m->config->N, m->config->col_dims, tmp_coil);
		linop_gdiag_set_diag(m->pattern, m->config->ND, m->config->pat_dims_merged, tmp_pattern);

		md_free(tmp_coil);
		md_free(tmp_pattern);
	}

	if (NULL != d->model->nufft) {

		linop_fmac_set_tensor(d->model->coils, d->model->config->N, d->model->config->col_dims, coil);

		nufft_update_psf(d->model->nufft, d->model->config->ND, d->model->config->psf_dims, pattern);
	}

	assert(NULL == d->model->nufft_loss);
}

static void sense_model_set_data_der(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	assert(0 == i);
	assert(0 == o);

	const auto d = CAST_DOWN(sense_model_set_data_s, _data);

	md_copy(d->N, d->dims, dst, src, CFL_SIZE);
}

static void sense_model_set_data_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(sense_model_set_data_s, _data);

	sense_model_free(d->model);

	xfree(d->dims);

	xfree(d);
}


/**
 * Returns: Update coils and pattern in linops (SENSE Operator)
 *
 * @param N
 * @param dims 	dummy dimensions for identity from input 0 to output 0
 * @param model SENSE model holding linops to update
 *
 * Input tensors:
 * dummy:	dims
 * coil:	col_dims (derived from model)
 * pattern:	pat_dims (derived from model)
 *
 * Output tensors:
 * dummy:	dims
 */
static const struct nlop_s* nlop_sense_model_set_data_create(int N, const long dims[N], struct sense_model_s* model, bool output_psf)
{
	PTR_ALLOC(struct sense_model_set_data_s, data);
	SET_TYPEID(sense_model_set_data_s, data);

	data->model = sense_model_ref(model);
	data->N = N;
	data->dims = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, data->dims, dims);

	data->output_psf = output_psf;

	int NM = MAX(N, model->config->ND);

	long nl_odims[2][NM];
	md_singleton_dims(NM, nl_odims[0]);
	md_singleton_dims(NM, nl_odims[1]);

	md_copy_dims(N, nl_odims[0], data->dims);
	md_copy_dims(model->config->ND, nl_odims[1], model->config->psf_dims);

	long nl_idims[3][NM];
	md_singleton_dims(NM, nl_idims[0]);
	md_singleton_dims(NM, nl_idims[1]);
	md_singleton_dims(NM, nl_idims[2]);

	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(model->config->N, nl_idims[1], model->config->col_dims);
	md_copy_dims(model->config->ND, nl_idims[2], model->config->psf_dims);

	nlop_der_fun_t der[3][output_psf ? 2 : 1];

	for(int i = 0; i< 3; i++)
		for (int o = 0; o < (output_psf ? 2 : 1); o++)
			der[i][o] = NULL;

	der[0][0] = sense_model_set_data_der;

	const struct nlop_s* result = nlop_generic_create(output_psf ? 2 : 1,
			NM, nl_odims, 3, NM, nl_idims, CAST_UP(PTR_PASS(data)),
			sense_model_set_data_fun, der, der,
			NULL, NULL, sense_model_set_data_del);

	result = nlop_reshape_out_F(result, 0, N, nl_odims[0]);

	if (output_psf)
		result = nlop_reshape_out_F(result, 1, model->config->ND, nl_odims[0]);

	result = nlop_reshape_in_F(result, 0, N, nl_idims[0]);
	result = nlop_reshape_in_F(result, 1, model->config->N, nl_idims[1]);
	result = nlop_reshape_in_F(result, 2, model->config->ND, nl_idims[2]);

	return result;
}


/**
 * Returns: Update coils and pattern in linops (SENSE Operator)
 *
 * @param N
 * @param dims 	dummy dimensions for identity from input 0 to output 0
 * @param model SENSE model holding linops to update
 *
 * Input tensors:
 * dummy:	dims
 * coil:	col_dims (derived from model)
 * pattern:	pat_dims (derived from model)
 *
 * Output tensors:
 * dummy:	dims
 */
const struct nlop_s* nlop_sense_model_set_data_batch_create(int N, const long dims[N], int Nb, struct sense_model_s* models[Nb])
{
	assert(N >= models[0]->config->N);
	assert(dims[BATCH_DIM] == Nb);

	long dims2[N];
	md_select_dims(N, ~BATCH_FLAG, dims2, dims);

	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_sense_model_set_data_create(N, dims2, models[i], false);

	if (1 == Nb)
		return nlops[0];

	int ostack_dim[] = { BATCH_DIM };
	int istack_dim[] = { BATCH_DIM, BATCH_DIM, BATCH_DIM };

	return nlop_stack_multiple_F(Nb, nlops, 3, istack_dim, 1, ostack_dim, true , multigpu);
}



static void sense_model_set_data_noncart_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(sense_model_set_data_s, _data);

	int i = 0;
	complex float* dst = args[i++];
	complex float* dst_psf = d->output_psf ? args[i++] : NULL;

	const complex float* src = args[i++];
	const complex float* coil = args[i++];
	const complex float* pattern = args[i++];
	const complex float* traj = args[i++];

	md_copy(d->N, d->dims, dst, src, CFL_SIZE);

	linop_fmac_set_tensor(d->model->coils, d->model->config->N, d->model->config->col_dims, coil);

	assert(NULL != d->model->nufft);

	nufft_update_traj(d->model->nufft, d->N, d->model->config->trj_dims, traj, d->model->config->pat_dims, pattern, d->model->config->bas_dims, d->model->config->basis);

	if (NULL != d->model->nufft_loss)
		nufft_update_traj(d->model->nufft_loss, d->N, d->model->config->trj_dims, traj, MD_SINGLETON_DIMS(d->N), NULL, d->model->config->bas_dims, d->model->config->basis);

	if (NULL != dst_psf)
		nufft_get_psf(d->model->nufft, d->model->config->ND, d->model->config->psf_dims, dst_psf);
}



/**
 * Returns: Update coils pattern and trajectory in linops (Noncart SENSE Operator)
 *
 * @param N
 * @param dims 	dummy dimensions for identity from input 0 to output 0
 * @param model SENSE model holding linops to update
 *
 * Input tensors:
 * dummy:	dims
 * coil:	col_dims (derived from model)
 * pattern:	wgh_dims (derived from model)
 * trajectory:	trj_dims (derived from model)
 *
 * Output tensors:
 * dummy:	dims
 */
static const struct nlop_s* nlop_sense_model_set_data_noncart_create(int N, const long dims[N], struct sense_model_s* model, bool output_psf)
{
	PTR_ALLOC(struct sense_model_set_data_s, data);
	SET_TYPEID(sense_model_set_data_s, data);

	data->model = sense_model_ref(model);
	data->N = N;
	data->dims = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, data->dims, dims);

	data->output_psf = output_psf;

	int NM = MAX(N, model->config->ND);

	long nl_odims[2][NM];
	md_singleton_dims(NM, nl_odims[0]);
	md_singleton_dims(NM, nl_odims[1]);

	md_copy_dims(N, nl_odims[0], data->dims);
	md_copy_dims(model->config->ND, nl_odims[1], model->config->psf_dims);

	long nl_idims[4][NM];
	md_singleton_dims(NM, nl_idims[0]);
	md_singleton_dims(NM, nl_idims[1]);
	md_singleton_dims(NM, nl_idims[2]);
	md_singleton_dims(NM, nl_idims[3]);

	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(model->config->N, nl_idims[1], model->config->col_dims);
	md_copy_dims(model->config->N, nl_idims[2], model->config->pat_dims);
	md_copy_dims(model->config->N, nl_idims[3], model->config->trj_dims);

	nlop_der_fun_t der[4][output_psf ? 2 : 1];

	for(int i = 0; i< 4; i++)
		for (int o = 0; o < (output_psf ? 2 : 1); o++)
			der[i][o] = NULL;

	der[0][0] = sense_model_set_data_der;


	const struct nlop_s* result = nlop_generic_create(output_psf ? 2 : 1,
			NM, nl_odims, 4, NM, nl_idims, CAST_UP(PTR_PASS(data)),
			sense_model_set_data_noncart_fun, der, der,
			NULL, NULL, sense_model_set_data_del);

	result = nlop_reshape_out_F(result, 0, N, nl_odims[0]);

	if (output_psf)
		result = nlop_reshape_out_F(result, 1, model->config->ND, nl_odims[1]);

	result = nlop_reshape_in_F(result, 0, N, nl_idims[0]);
	result = nlop_reshape_in_F(result, 1, model->config->N, nl_idims[1]);
	result = nlop_reshape_in_F(result, 2, model->config->N, nl_idims[2]);
	result = nlop_reshape_in_F(result, 3, model->config->N, nl_idims[3]);

	return result;
}


/**
 * Returns: Adjoint SENSE model
 *
 * @param model
 *
 * Input tensors:
 * kspace:	ksp_dims (derived from model)
 * coil:	col_dims (derived from model)
 * pattern:	wgh_dims (derived from model)
 * [trajectory:	trj_dims (derived from model)]
 *
 * Output tensors:
 * image:	img_dims (derived from model)
 */
const struct nlop_s* nlop_sense_adjoint_create(int Nb, struct sense_model_s* models[Nb], bool output_psf)
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++) {

		nlops[i] = nlop_from_linop_F(linop_get_adjoint(models[i]->sense));

		if (NULL == models[i]->nufft)
			nlops[i] = nlop_chain2_FF(nlop_sense_model_set_data_create(models[i]->config->N, models[i]->config->ksp_dims, models[i], output_psf), 0, nlops[i], 0);
		else
			nlops[i] = nlop_chain2_FF(nlop_sense_model_set_data_noncart_create(models[i]->config->N, models[i]->config->ksp_dims, models[i], output_psf), 0, nlops[i], 0);
	}

	if (1 == Nb)
		return nlops[0];

	int ostack_dim[] = { BATCH_DIM, BATCH_DIM };
	int istack_dim[] = { BATCH_DIM, BATCH_DIM, BATCH_DIM, BATCH_DIM };

	return nlop_stack_multiple_F(Nb, nlops, (NULL == models[0]->nufft) ? 3 : 4, istack_dim, output_psf ? 2 : 1, ostack_dim, true , multigpu);
}

const struct nlop_s* nlop_sense_normal_create(int Nb, struct sense_model_s* models[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_from_linop_F(linop_get_normal(models[i]->sense));

	if (1 == Nb)
		return nlops[0];

	int ostack_dim[] = { BATCH_DIM };
	int istack_dim[] = { BATCH_DIM };

	return nlop_stack_multiple_F(Nb, nlops, 1, istack_dim, 1, ostack_dim, true, multigpu);
}

const struct nlop_s* nlop_sense_normal_inv_create(int Nb, struct sense_model_s* models[Nb], struct iter_conjgrad_conf* iter_conf, unsigned long lambda_flags)
{
	struct nlop_norm_inv_conf norm_inv_conf = {

		.store_nlop = true,
		.iter_conf = iter_conf,
	};

	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = norm_inv_lop_lambda_create(&norm_inv_conf, models[i]->sense, lambda_flags);

	if (1 == Nb)
		return nlops[0];

	int ostack_dim[] = { BATCH_DIM };
	int istack_dim[] = { BATCH_DIM, MD_IS_SET(lambda_flags, BATCH_DIM) ? BATCH_DIM : -1 };

	return nlop_stack_multiple_F(Nb, nlops, 2, istack_dim, 1, ostack_dim, true, multigpu);
}

const struct nlop_s* nlop_sense_dc_prox_create(int Nb, struct sense_model_s* models[Nb], struct iter_conjgrad_conf* iter_conf, unsigned long lambda_flags)
{
	auto result = nlop_sense_normal_inv_create(Nb, models, iter_conf, lambda_flags);

	int N = models[0]->config->N;
	long img_dims[N];
	long lam_dims[N];

	md_copy_dims(N, img_dims, nlop_generic_domain(result, 0)->dims);
	md_copy_dims(N, lam_dims, nlop_generic_domain(result, 1)->dims);

	result = nlop_chain2_swap_FF(nlop_zaxpbz_create(N, img_dims, 1, 1), 0, result, 0);
	result = nlop_chain2_swap_FF(nlop_tenmul_create(N, img_dims, img_dims, lam_dims), 0, result, 0);
	result = nlop_dup_F(result, 1, 3);
	result = nlop_shift_input_F(result, 2, 1);

	return result;
}

const struct nlop_s* nlop_sense_dc_grad_create(int Nb, struct sense_model_s* models[Nb], unsigned long lambda_flags)
{
	auto result = nlop_sense_normal_create(Nb, models);

	int N = models[0]->config->N;
	long img_dims[N];
	long lam_dims[N];

	md_copy_dims(N, img_dims, nlop_generic_domain(result, 0)->dims);
	md_select_dims(N, lambda_flags, lam_dims, img_dims);

	result = nlop_chain2_swap_FF(result, 0, nlop_zaxpbz_create(N, img_dims, 1., -1.), 0);
	result = nlop_chain2_swap_FF(result, 0, nlop_tenmul_create(N, img_dims, img_dims, lam_dims), 0);

	return result;
}

const struct nlop_s* nlop_sense_scale_maxeigen_create(int Nb, struct sense_model_s* models[Nb], int N, const long dims[N])
{
	assert(N >= models[0]->config->N);

	long dims_scl[N];
	md_select_dims(N, ~BATCH_FLAG, dims_scl, dims);

	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++) {

		auto normal_op = nlop_from_linop_F(linop_get_normal(models[i]->sense));
		normal_op = nlop_combine_FF(normal_op, nlop_del_out_create(N, dims_scl)); // that is necessary to apply operators in correct order
		nlops[i] = nlop_maxeigen_create(normal_op);
		nlops[i] = nlop_no_der_F(nlops[i], 0, 0);
		nlop_free(normal_op);
	}

	int ostack_dim[] = { BATCH_DIM };
	int istack_dim[] = { BATCH_DIM };

	const struct nlop_s* result = NULL;

	if (1 == Nb)
		result = nlops[0];
	else
		result = nlop_stack_multiple_F(Nb, nlops, 1, istack_dim, 1, ostack_dim, true , multigpu);

	long odims[N];
	md_select_dims(N, BATCH_FLAG, odims, dims);

	result = nlop_reshape_out_F(result, 0, N, odims);
	result = nlop_chain_FF(result, nlop_zinv_create(N, odims));
	result = nlop_chain2_FF(result, 0, nlop_tenmul_create(N, dims, dims, odims), 1);
	result = nlop_dup_F(result, 0, 1);

	return result;
}


static const struct nlop_s* nlop_mri_normal_slice_create(const struct config_nlop_mri_s* conf)
{
	assert(NULL != conf);

	struct sense_model_s* model = sense_model_normal_create(conf);

	auto result = nlop_sense_model_set_data_create(conf->N, conf->img_dims, model, false);

	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(linop_get_normal(model->sense)), 0);

	sense_model_free(model);

	return result;
}


static const struct nlop_s* nlop_mri_loss_create_s(bool fft, struct sense_model_s* model)
{
	const struct linop_s* lop_sense = linop_clone(model->coils);

	if (fft) {

		if (model->config->noncart && NULL == model->nufft_loss) {

			struct nufft_conf_s conf = model->config->nufft_conf;
			conf.toeplitz = false;
			model->nufft_loss = nufft_create2(model->config->N, model->config->ksp_dims, model->config->cim_dims, model->config->trj_dims, NULL, MD_SINGLETON_DIMS(model->config->N), NULL, model->config->bas_dims, model->config->basis, conf);
		}

		if (model->config->noncart)
			lop_sense = linop_chain_FF(lop_sense, linop_clone(model->nufft_loss));
		else
			lop_sense = linop_chain_FF(linop_chain_FF(lop_sense, linop_fft_create(model->config->N, model->config->ksp_dims, FFT_FLAGS)),
							linop_cdiag_create(model->config->N, model->config->ksp_dims, md_nontriv_dims(model->config->N, model->config->fftmod_precomp_ksp_dims), multiplace_read(model->config->fftmod_precomp_ksp, NULL)));
	} else {

		if (!model->config->noncart) {

			const struct linop_s* lop_fftmod = linop_cdiag_create(model->config->N, model->config->cim_dims, md_nontriv_dims(model->config->N, model->config->fftmod_precomp_img_dims), multiplace_read(model->config->fftmod_precomp_img, NULL));

			lop_sense = linop_chain_FF(lop_sense, linop_get_adjoint(lop_fftmod));
			lop_sense = linop_chain_FF(lop_sense, linop_scale_create(model->config->N, model->config->cim_dims, sqrtf(md_calc_size(3, model->config->cim_dims))));

			linop_free(lop_fftmod);
		}

	}

	return  nlop_from_linop_F(lop_sense);
}

const struct nlop_s* nlop_mri_loss_create(bool fft, int Nb, struct sense_model_s* model[Nb])
{
	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_mri_loss_create_s(fft, model[i]);

	int istack_dims[1] = { BATCH_DIM };
	int ostack_dims[1] = { BATCH_DIM };

	const struct nlop_s* ret = nlop_stack_multiple_F(Nb, nlops, 1, istack_dims, 1, ostack_dims, true, multigpu);

	return ret;
}


/**
 * Returns: MRI normal operator
 *
 * @param conf
 *
 *
 * for default dims:
 *
 * Input tensors:
 * image:	img_dims: 	(Nx, Ny, Nz, 1,  ..., Nb )
 * coil:	col_dims:	(Nx, Ny, Nz, Nc, ..., Nb )
 * pattern:	pat_dims:	(Nx, Ny, Nz, 1,  ..., Nb )
 *
 * Output tensors:
 * image:	img_dims: 	(Nx, Ny, Nz, 1,  ..., Nb )
 */

const struct nlop_s* nlop_mri_normal_create(int Nb, const struct config_nlop_mri_s* conf)
{
	if (1 == Nb)
		return nlop_mri_normal_slice_create(conf);

	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_mri_normal_slice_create(conf);

	int ostack_dim[] = { BATCH_DIM };
	int istack_dim[] = { BATCH_DIM, BATCH_DIM, BATCH_DIM };

	return nlop_stack_multiple_F(Nb, nlops, 3, istack_dim, 1, ostack_dim, true , multigpu);
}




static const struct nlop_s* nlop_mri_normal_inv_slice_create(int N, const long lam_dims[N], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	assert(NULL != conf);

	struct sense_model_s* model = sense_model_normal_create(conf);

	assert(N == conf->N);
	auto result = nlop_sense_model_set_data_create(N, conf->img_dims, model, false);

	struct nlop_norm_inv_conf norm_inv_conf = {

		.store_nlop = true,
		.iter_conf = iter_conf,
	};

	result = nlop_chain2_FF(result, 0, norm_inv_lop_lambda_create(&norm_inv_conf, model->sense, md_nontriv_dims(N, lam_dims)), 0);
	result = nlop_shift_input_F(result, 3, 0);

	sense_model_free(model);

	return result;
}



/**
 * Create an operator applying the inverse normal mri forward model on its input
 *
 * out = (A^HA +l1)^-1 in
 * A = Pattern FFT Coils
 *
 * @param N
 * @param lam_dims
 * @param conf
 * @param iter_conf configuration for conjugate gradient
 *
 * for default dims:
 *
 * Input tensors:
 * image:	img_dims: 	(Nx, Ny, Nz, 1,  ..., Nb )
 * coil:	col_dims:	(Nx, Ny, Nz, Nc, ..., Nb )
 * pattern:	pat_dims:	(Nx, Ny, Nz, 1,  ..., Nb )
 * lambda:	lam_dims:	(1 , 1 , 1 , 1,  ..., Nb )
 *
 * Output tensors:
 * image:	img_dims: 	(Nx, Ny, Nz, 1,  ..., Nb )
 */
const struct nlop_s* nlop_mri_normal_inv_create(int N, const long lam_dims[N], int Nb, const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	if (1 == Nb)
		return nlop_mri_normal_inv_slice_create(N, lam_dims, conf, iter_conf);

	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_mri_normal_inv_slice_create(N, lam_dims, conf, iter_conf);

	int ostack_dim[] = { BATCH_DIM };
	int istack_dim[] = { BATCH_DIM, BATCH_DIM, BATCH_DIM, (1 == lam_dims[BATCH_DIM]) ? -1 : BATCH_DIM };

	return nlop_stack_multiple_F(Nb, nlops, 4, istack_dim, 1, ostack_dim, true , multigpu);
}



/**
 * Create an operator minimizing the following functional
 *
 * out = argmin 0.5 ||Ax-y||_2^2 + 0.5 ||sqrt{lambda} (x-x_0)||_2^2
 * A = Pattern FFT Coils
 *
 * @param N
 * @param max_dims
 * @param lam_dims
 * @param ND
 * @param psf_dims
 * @param conf can be NULL to fallback on nlop_mri_simple
 * @param iter_conf configuration for conjugate gradient
 *
 * Input tensors:
 * x0:		idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * adjoint:	idims: 	(Nx, Ny, Nz,  1, ..., Nb)
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1, ..., 1 / Nb)
 * lambda:	ldims:	( 1,  1,  1,  1, ..., Nb)
 *
 * Output tensors:
 * image:	idims: 	(Nx, Ny, Nz, 1, ..., Nb)
 */
const struct nlop_s* nlop_mri_dc_prox_create(int N, const long lam_dims[N], int Nb, const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	auto result = nlop_mri_normal_inv_create(N, lam_dims, Nb, conf, iter_conf);

	long img_dims[N];
	md_copy_dims(N, img_dims, nlop_generic_codomain(result, 0)->dims);

	result = nlop_chain2_swap_FF(nlop_zaxpbz_create(N, img_dims, 1., 1.), 0, result, 0); //in: lambda*x0, AHy, coil, pattern, lambda
	result = nlop_chain2_swap_FF(nlop_tenmul_create(N, img_dims, img_dims, lam_dims),0 , result, 0); //in: x0, lambda, AHy, coil, pattern, lambda
	result = nlop_dup_F(result, 1, 5); //in: x0, lambda, AHy, coil, pattern
	result = nlop_shift_input_F(result, 4, 1); //in: x0, AHy, coil, pattern, lambda
	result = nlop_chain2_FF(nlop_from_linop_F(linop_zreal_create(N, lam_dims)), 0, result, 4);

	return result;
}



static const struct nlop_s* nlop_mri_normal_max_eigen_slice_create(const struct config_nlop_mri_s* conf)
{
	assert(NULL != conf);

	struct sense_model_s* model = sense_model_normal_create(conf);

	complex float zero = 0;

	auto result = nlop_sense_model_set_data_create(1, MD_DIMS(1), model, false);

	result = nlop_set_input_const_F(result, 0, 1, MD_DIMS(1), true, &zero);

	auto normal_op = nlop_from_linop_F(linop_get_normal(model->sense));

	normal_op = nlop_combine_FF(normal_op, nlop_del_out_create(1, MD_DIMS(1))); // that is necessary to apply operators in correct order

	result = nlop_chain2_FF(result, 0, nlop_maxeigen_create(normal_op), 0);

	sense_model_free(model);

	return result;
}


/**
 * Returns: Operator computing max eigen value of SENSE operator
 *
 * @param N
 * @param max_dims
 * @param ND
 * @param psf_dims pattern (Cartesian) or psf (Non-Cartesian)
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * coil:	cdims:	(Nx, Ny, Nz, Nc, ..., Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  ..., 1 )
 *
 * Output tensors:
 * max eigen:	ldims: 	( 1,  1,  1,  1, ..., Nb)
 */

const struct nlop_s* nlop_mri_normal_max_eigen_create(int Nb, const struct config_nlop_mri_s* conf)
{
	if (1 == Nb)
		return nlop_mri_normal_max_eigen_slice_create(conf);

	const struct nlop_s* nlops[Nb];

	for (int i = 0; i < Nb; i++)
		nlops[i] = nlop_mri_normal_max_eigen_slice_create(conf);

	int ostack_dim[] = { BATCH_DIM };
	int istack_dim[] = { BATCH_DIM, BATCH_DIM, BATCH_DIM };

	return nlop_stack_multiple_F(Nb, nlops, 2, istack_dim, 1, ostack_dim, true , multigpu);
}




struct mri_scale_rss_s {

	nlop_data_t super;
	int N;

	unsigned long rss_flag;
	unsigned long bat_flag;

	const long* col_dims;

	bool mean;
};

DEF_TYPEID(mri_scale_rss_s);

static void mri_scale_rss_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(mri_scale_rss_s, _data);

	int N = d->N;

	md_zrss(N, d->col_dims, d->rss_flag, dst, src);

	if (d->mean) {

		long bdims[N];
		long idims[N];
		md_select_dims(N, d->bat_flag, bdims, d->col_dims);
		md_select_dims(N, ~d->rss_flag, idims, d->col_dims);

		complex float* mean = md_alloc_sameplace(N, bdims, CFL_SIZE, dst);
		complex float* ones = md_alloc_sameplace(N, bdims, CFL_SIZE, dst);

		md_zfill(N, bdims, ones, 1.);

		md_zsum(N, idims, ~d->bat_flag, mean, dst);
		md_zsmul(N, bdims, mean, mean, (float)md_calc_size(N, bdims) / (float)md_calc_size(N, idims));
		md_zdiv(N, bdims, mean, ones, mean);

		md_zmul2(N, idims, MD_STRIDES(N, idims, CFL_SIZE), dst, MD_STRIDES(N, idims, CFL_SIZE), dst, MD_STRIDES(N, bdims, CFL_SIZE), mean);

		md_free(mean);
		md_free(ones);
	}
}

static void mri_scale_rss_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_scale_rss_s, _data);

	xfree(d->col_dims);
	xfree(d);
}

const struct nlop_s* nlop_mri_scale_rss_create(int Nb, const struct config_nlop_mri_s* conf)
{
	PTR_ALLOC(struct mri_scale_rss_s, data);
	SET_TYPEID(mri_scale_rss_s, data);

	PTR_ALLOC(long[conf->N], col_dims);

	int N = conf->N;
	assert(DIMS == N);

	md_copy_dims(conf->N, *col_dims, conf->col_dims);

	data->col_dims = *PTR_PASS(col_dims);

	data->N = conf->N;
	data->bat_flag = BATCH_FLAG;
	data->rss_flag = (~md_nontriv_dims(conf->N, conf->img_dims)) & (md_nontriv_dims(conf->N, conf->col_dims));
	data->mean = true;

	long odims[N];
	long idims[N];
	md_copy_dims(N, idims, conf->cim_dims);
	md_copy_dims(N, odims, conf->img_dims);

	odims[BATCH_DIM] = Nb;
	idims[BATCH_DIM] = Nb;

	return nlop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)), mri_scale_rss_fun, NULL, NULL, NULL, NULL, mri_scale_rss_del);
}

