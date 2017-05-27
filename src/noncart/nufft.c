/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Frank Ong <frankong@berkeley.edu>
 * 2014-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * Strang G. A proposal for Toeplitz matrix calculations. Journal Studies in Applied Math. 1986; 74(2):171-17
 *
 */

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/fft.h"
#include "num/shuffle.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"
#include "linops/someops.h"

#include "noncart/grid.h"

#include "nufft.h"

#define FFT_FLAGS (MD_BIT(0)|MD_BIT(1)|MD_BIT(2))

struct nufft_conf_s nufft_conf_defaults = {

	.toeplitz = false,
	.pcycle = false,
};


/**
 *
 * NUFFT internal data structure
 *
 */
struct nufft_data {

	INTERFACE(linop_data_t);

	struct nufft_conf_s conf;	///< NUFFT configuration structure

	unsigned int N;			///< Number of dimension

	const complex float* linphase;	///< Linear phase for pruned FFT
	const complex float* traj;	///< Trajectory
	const complex float* roll;	///< Roll-off factor
	const complex float* psf;	///< Point-spread function (2x size)
	const complex float* fftmod;	///< FFT modulation for centering
	const complex float* weights;	///< Weights, ex, density compensation
#ifdef USE_CUDA
	const complex float* linphase_gpu;
	const complex float* psf_gpu;
	complex float* grid_gpu;
#endif
	complex float* grid;		///< Oversampling grid

	float width;			///< Interpolation kernel width
	double beta;			///< Kaiser-Bessel beta parameter

	const struct linop_s* fft_op;	///< FFT operator

	long* ksp_dims;			///< Kspace dimension
	long* cim_dims;			///< Coil image dimension
	long* cml_dims;			///< Coil + linear phase dimension
	long* img_dims;			///< Image dimension
	long* trj_dims;			///< Trajectory dimension
	long* lph_dims;			///< Linear phase dimension
	long* psf_dims;			///< Point spread function dimension
	long* wgh_dims;			///< Weights dimension

	//!
	long* cm2_dims;			///< 2x oversampled coil image dimension

	long* ksp_strs;
	long* cim_strs;
	long* cml_strs;
	long* img_strs;
	long* trj_strs;
	long* lph_strs;
	long* psf_strs;
	long* wgh_strs;

	const struct linop_s* cfft_op;   ///< Pcycle FFT operator
	unsigned int cycle;
	long* rcml_dims;		///< Pcycle Coil + linear phase dimension
	long* rlph_dims;		///< Pcycle Linear phase dimension
	long* rpsf_dims;		///< Pcycle Point spread function dimension
	
	long* rcml_strs;		
	long* rlph_strs;		
	long* rpsf_strs;		
	
};

DEF_TYPEID(nufft_data);


static void nufft_free_data(const linop_data_t* data);
static void nufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src);
static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src);


static void toeplitz_mult(const struct nufft_data* data, complex float* dst, const complex float* src);
static void toeplitz_mult_pcycle(const struct nufft_data* data, complex float* dst, const complex float* src);
static complex float* compute_linphases(unsigned int N, long lph_dims[N + 3], const long img_dims[N]);
static complex float* compute_psf2(unsigned int N, const long psf_dims[N + 3], const long trj_dims[N], const complex float* traj, const complex float* weights);


/**
 * NUFFT operator initialization
 */
struct linop_s* nufft_create(unsigned int N,			///< Number of dimension
			     const long ksp_dims[N],		///< kspace dimension
			     const long cim_dims[N],		///< Coil images dimension
			     const long traj_dims[N],		///< Trajectory dimension
			     const complex float* traj,		///< Trajectory
			     const complex float* weights,	///< Weights, ex, soft-gating or density compensation
			     struct nufft_conf_s conf)		///< NUFFT configuration options

{
	PTR_ALLOC(struct nufft_data, data);
	SET_TYPEID(nufft_data, data);

	data->N = N;
	data->traj = traj;
	data->conf = conf;

	data->width = 3.;
	data->beta = calc_beta(2., data->width);

	// get dims

	assert(md_check_compat(N - 3, 0, ksp_dims + 3, cim_dims + 3));

	unsigned int ND = N + 3;

	data->ksp_dims = *TYPE_ALLOC(long[ND]);
	data->cim_dims = *TYPE_ALLOC(long[ND]);
	data->cml_dims = *TYPE_ALLOC(long[ND]);
	data->img_dims = *TYPE_ALLOC(long[ND]);
	data->trj_dims = *TYPE_ALLOC(long[ND]);
	data->lph_dims = *TYPE_ALLOC(long[ND]);
	data->psf_dims = *TYPE_ALLOC(long[ND]);
	data->wgh_dims = *TYPE_ALLOC(long[ND]);

	data->ksp_strs = *TYPE_ALLOC(long[ND]);
	data->cim_strs = *TYPE_ALLOC(long[ND]);
	data->cml_strs = *TYPE_ALLOC(long[ND]);
	data->img_strs = *TYPE_ALLOC(long[ND]);
	data->trj_strs = *TYPE_ALLOC(long[ND]);
	data->lph_strs = *TYPE_ALLOC(long[ND]);
	data->psf_strs = *TYPE_ALLOC(long[ND]);
	data->wgh_strs = *TYPE_ALLOC(long[ND]);
	
	data->rlph_dims = *TYPE_ALLOC(long[ND]);
	data->rpsf_dims = *TYPE_ALLOC(long[ND]);
	data->rcml_dims = *TYPE_ALLOC(long[ND]);
	data->rlph_strs = *TYPE_ALLOC(long[ND]);
	data->rpsf_strs = *TYPE_ALLOC(long[ND]);
	data->rcml_strs = *TYPE_ALLOC(long[ND]);

	md_singleton_dims(ND, data->cim_dims);
	md_singleton_dims(ND, data->ksp_dims);

	md_copy_dims(N, data->cim_dims, cim_dims);
	md_copy_dims(N, data->ksp_dims, ksp_dims);


	md_select_dims(ND, FFT_FLAGS, data->img_dims, data->cim_dims);

	assert(3 == traj_dims[0]);
	assert(traj_dims[1] == ksp_dims[1]);
	assert(traj_dims[2] == ksp_dims[2]);
	assert(md_check_compat(N - 3, ~0, traj_dims + 3, ksp_dims + 3));
	assert(md_check_bounds(N - 3, ~0, traj_dims + 3, ksp_dims + 3));

	md_singleton_dims(ND, data->trj_dims);
	md_copy_dims(N, data->trj_dims, traj_dims);


	// get strides

	md_calc_strides(ND, data->cim_strs, data->cim_dims, CFL_SIZE);
	md_calc_strides(ND, data->img_strs, data->img_dims, CFL_SIZE);
	md_calc_strides(ND, data->trj_strs, data->trj_dims, CFL_SIZE);
	md_calc_strides(ND, data->ksp_strs, data->ksp_dims, CFL_SIZE);


	data->weights = NULL;

	if (NULL != weights) {

		md_singleton_dims(ND, data->wgh_dims);
		md_select_dims(N, ~MD_BIT(0), data->wgh_dims, data->trj_dims);
		md_calc_strides(ND, data->wgh_strs, data->wgh_dims, CFL_SIZE);

		complex float* tmp = md_alloc(ND, data->wgh_dims, CFL_SIZE);
		md_copy(ND, data->wgh_dims, tmp, weights, CFL_SIZE);
		data->weights = tmp;
	}


	complex float* roll = md_alloc(ND, data->img_dims, CFL_SIZE);
	rolloff_correction(2., data->width, data->beta, data->img_dims, roll);
	data->roll = roll;


	complex float* linphase = compute_linphases(N, data->lph_dims, data->img_dims);

	md_calc_strides(ND, data->lph_strs, data->lph_dims, CFL_SIZE);

	if (!conf.toeplitz)
		md_zmul2(ND, data->lph_dims, data->lph_strs, linphase, data->lph_strs, linphase, data->img_strs, data->roll);


	fftmod(ND, data->lph_dims, FFT_FLAGS, linphase, linphase);
	fftscale(ND, data->lph_dims, FFT_FLAGS, linphase, linphase);

	float scale = 1.;
	for (unsigned int i = 0; i < N; i++)
		scale *= ((data->lph_dims[i] > 1) && (i < 3)) ? 0.5 : 1.;

	md_zsmul(ND, data->lph_dims, linphase, linphase, scale);


	complex float* fftm = md_alloc(ND, data->img_dims, CFL_SIZE);
	md_zfill(ND, data->img_dims, fftm, 1.);
	fftmod(ND, data->img_dims, FFT_FLAGS, fftm, fftm);
	data->fftmod = fftm;



	data->linphase = linphase;
	data->psf = NULL;
#ifdef USE_CUDA
	data->linphase_gpu = NULL;
	data->psf_gpu = NULL;
	data->grid_gpu = NULL;
#endif
	if (conf.toeplitz) {

		debug_printf(DP_DEBUG1, "NUFFT: Toeplitz mode\n");

#if 0
		md_copy_dims(ND, data->psf_dims, data->lph_dims);
#else
		md_copy_dims(3, data->psf_dims, data->lph_dims);
		md_copy_dims(ND - 3, data->psf_dims + 3, data->trj_dims + 3);
		data->psf_dims[N] = data->lph_dims[N];
#endif
		md_calc_strides(ND, data->psf_strs, data->psf_dims, CFL_SIZE);
		data->psf = compute_psf2(N, data->psf_dims, data->trj_dims, data->traj, data->weights);
	}


	md_copy_dims(ND, data->cml_dims, data->cim_dims);
	data->cml_dims[N + 0] = data->lph_dims[N + 0];

	md_calc_strides(ND, data->cml_strs, data->cml_dims, CFL_SIZE);


	data->cm2_dims = *TYPE_ALLOC(long[ND]);
	// !
	md_copy_dims(ND, data->cm2_dims, data->cim_dims);

	for (int i = 0; i < 3; i++)
		data->cm2_dims[i] = (1 == cim_dims[i]) ? 1 : (2 * cim_dims[i]);



	data->grid = md_alloc(ND, data->cml_dims, CFL_SIZE);

	data->fft_op = linop_fft_create(ND, data->cml_dims, FFT_FLAGS);

	if (conf.pcycle) {

		debug_printf(DP_DEBUG1, "NUFFT: Pcycle Mode\n");
		data->cycle = 0;
		data->cfft_op = linop_fft_create(N, data->cim_dims, FFT_FLAGS);
	}


	return linop_create(N, ksp_dims, N, cim_dims,
			CAST_UP(PTR_PASS(data)), nufft_apply, nufft_apply_adjoint, nufft_apply_normal, NULL, nufft_free_data);
}



/**
 * Compute Strang's circulant preconditioner
 *
 * Strang's reconditioner is simply the cropped psf in the image domain
 *
 * Ref: Strang G. A proposal for Toeplitz matrix calculations. Journal Studies in Applied Math. 1986; 74(2):171-17
 */
static complex float* compute_precond(unsigned int N, const long* pre_dims, const long* pre_strs, const long* psf_dims, const long* psf_strs, const complex float* psf, const complex float* linphase)
{
	unsigned int ND = N + 3;

	complex float* pre = md_alloc(ND, pre_dims, CFL_SIZE);
	complex float* psft = md_alloc(ND, psf_dims, CFL_SIZE);

	// Transform psf to image domain
	ifftuc(ND, psf_dims, FFT_FLAGS, psft, psf);

	// Compensate for linear phase to get cropped psf
	md_clear(ND, pre_dims, pre, CFL_SIZE);
	md_zfmacc2(ND, psf_dims, pre_strs, pre, psf_strs, psft, psf_strs, linphase);
	
        md_free(psft);
        
	// Transform to Fourier domain
	fftuc(N, pre_dims, FFT_FLAGS, pre, pre);

	for(int i = 0; i < md_calc_size( N, pre_dims ); i++)
		pre[i] = cabsf(pre[i]);
	
	md_zsadd(N, pre_dims, pre, pre, 1e-3);
	
	return pre;
}



/**
 * NUFFT precondition internal data structure
 */
struct nufft_precond_data {

	INTERFACE(operator_data_t);

	unsigned int N;
	const complex float* pre; ///< Preconditioner

	long* cim_dims; ///< Coil image dimension
	long* pre_dims; ///< Preconditioner dimension

	long* cim_strs;
	long* pre_strs;

	const struct linop_s* fft_op; ///< FFT linear operator
};


DEF_TYPEID(nufft_precond_data);


static void nufft_precond_apply(const operator_data_t* _data, unsigned int M, void* args[M])
{
	assert(2 == M);

	const struct nufft_precond_data* data = CAST_DOWN(nufft_precond_data, _data);

	complex float* dst = args[0];
	const complex float* src = args[1];

	linop_forward(data->fft_op, data->N, data->cim_dims, dst, data->N, data->cim_dims, src);

	md_zdiv2(data->N, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst, data->pre_strs, data->pre);
	linop_adjoint(data->fft_op, data->N, data->cim_dims, dst, data->N, data->cim_dims, dst);
}

static void nufft_precond_del(const operator_data_t* _data)
{
	const struct nufft_precond_data* data = CAST_DOWN(nufft_precond_data, _data);

	xfree(data->cim_dims);
	xfree(data->pre_dims);
	xfree(data->cim_strs);
	xfree(data->pre_strs);

	md_free(data->pre);
	xfree(data);
}

const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op)
{
	const struct nufft_data* data = CAST_DOWN(nufft_data, linop_get_data(nufft_op));

	PTR_ALLOC(struct nufft_precond_data, pdata);
	SET_TYPEID(nufft_precond_data, pdata);

	assert(data->conf.toeplitz);

	unsigned int N = data->N;
	unsigned int ND = N + 3;

	pdata->N = N;
	pdata->cim_dims = *TYPE_ALLOC(long[ND]);
	pdata->pre_dims = *TYPE_ALLOC(long[ND]);
	pdata->cim_strs = *TYPE_ALLOC(long[ND]);
	pdata->pre_strs = *TYPE_ALLOC(long[ND]);

	md_copy_dims(ND, pdata->cim_dims, data->cim_dims);
	md_select_dims(ND, FFT_FLAGS, pdata->pre_dims, pdata->cim_dims);

	md_calc_strides(ND, pdata->cim_strs, pdata->cim_dims, CFL_SIZE);
	md_calc_strides(ND, pdata->pre_strs, pdata->pre_dims, CFL_SIZE);

	pdata->pre = compute_precond(pdata->N, pdata->pre_dims, pdata->pre_strs, data->psf_dims, data->psf_strs, data->psf, data->linphase);

	pdata->fft_op = linop_fft_create(pdata->N, pdata->cim_dims, FFT_FLAGS);

	const long* cim_dims = pdata->cim_dims;	// need to dereference pdata before PTR_PASS

	return operator_create(N, cim_dims, N, cim_dims, CAST_UP(PTR_PASS(pdata)), nufft_precond_apply, nufft_precond_del);
}



static complex float* compute_linphases(unsigned int N, long lph_dims[N + 3], const long img_dims[N + 3])
{
	float shifts[8][3];

	int s = 0;
	for(int i = 0; i < 8; i++) {

		bool skip = false;

		for(int j = 0; j < 3; j++) {

			shifts[s][j] = 0.;

			if (MD_IS_SET(i, j)) {

				skip = skip || (1 == img_dims[j]);
				shifts[s][j] = -0.5;
			}
		}

		if (!skip)
			s++;
	}

	unsigned int ND = N + 3;
	md_select_dims(ND, FFT_FLAGS, lph_dims, img_dims);
	lph_dims[N + 0] = s;

	complex float* linphase = md_alloc(ND, lph_dims, CFL_SIZE);

	for(int i = 0; i < s; i++) {

		float shifts2[ND];
		for (unsigned int j = 0; j < ND; j++)
			shifts2[j] = 0.;

		shifts2[0] = shifts[i][0];
		shifts2[1] = shifts[i][1];
		shifts2[2] = shifts[i][2];

		linear_phase(ND, img_dims, shifts2, 
				linphase + i * md_calc_size(ND, img_dims));
	}

	return linphase;
}


complex float* compute_psf(unsigned int N, const long img2_dims[N], const long trj_dims[N], const complex float* traj, const complex float* weights)
{      
	long ksp_dims1[N];
	md_select_dims(N, ~MD_BIT(0), ksp_dims1, trj_dims);

	struct linop_s* op2 = nufft_create(N, ksp_dims1, img2_dims, trj_dims, traj, NULL, nufft_conf_defaults);

	complex float* ones = md_alloc(N, ksp_dims1, CFL_SIZE);
	md_zfill(N, ksp_dims1, ones, 1.);

	if (NULL != weights) {

		md_zmul(N, ksp_dims1, ones, ones, weights);
		md_zmulc(N, ksp_dims1, ones, ones, weights);
	}

	complex float* psft = md_alloc(N, img2_dims, CFL_SIZE);

	linop_adjoint_unchecked(op2, psft, ones);

	md_free(ones);
	linop_free(op2);

	return psft;
}


static complex float* compute_psf2(unsigned int N, const long psf_dims[N + 3], const long trj_dims[N + 3], const complex float* traj, const complex float* weights)
{
	unsigned int ND = N + 3;

	long img_dims[ND];
	long img_strs[ND];

	md_select_dims(ND, ~MD_BIT(N + 0), img_dims, psf_dims);
	md_calc_strides(ND, img_strs, img_dims, CFL_SIZE);

	// PSF 2x size

	long img2_dims[ND];
	long img2_strs[ND];

	md_copy_dims(ND, img2_dims, img_dims);

	for (int i = 0; i < 3; i++)
		img2_dims[i] = (1 == img_dims[i]) ? 1 : (2 * img_dims[i]);

	md_calc_strides(ND, img2_strs, img2_dims, CFL_SIZE);

	complex float* traj2 = md_alloc(ND, trj_dims, CFL_SIZE);
	md_zsmul(ND, trj_dims, traj2, traj, 2.);

	complex float* psft = compute_psf(ND, img2_dims, trj_dims, traj2, weights);
	md_free(traj2);

	fftuc(ND, img2_dims, FFT_FLAGS, psft, psft);

	float scale = 1.;
	for (unsigned int i = 0; i < N; i++)
		scale *= ((img2_dims[i] > 1) && (i < 3)) ? 4. : 1.;

	md_zsmul(ND, img2_dims, psft, psft, scale);

	// reformat

	complex float* psf = md_alloc(ND, psf_dims, CFL_SIZE);

	long factors[N];

	for (unsigned int i = 0; i < N; i++)
		factors[i] = ((img_dims[i] > 1) && (i < 3)) ? 2 : 1;

	md_decompose(N + 0, factors, psf_dims, psf, img2_dims, psft, CFL_SIZE);

	md_free(psft);
	return psf;
}





static void nufft_free_data(const linop_data_t* _data)
{
	struct nufft_data* data = CAST_DOWN(nufft_data, _data);

	free(data->ksp_dims);
	free(data->cim_dims);
	free(data->cml_dims);
	free(data->img_dims);
	free(data->trj_dims);
	free(data->lph_dims);
	free(data->psf_dims);
	free(data->wgh_dims);

	free(data->ksp_strs);
	free(data->cim_strs);
	free(data->cml_strs);
	free(data->img_strs);
	free(data->trj_strs);
	free(data->lph_strs);
	free(data->psf_strs);
	free(data->wgh_strs);
	
	md_free(data->grid);
	md_free(data->linphase);
	md_free(data->psf);
	md_free(data->fftmod);
	md_free(data->weights);

#ifdef USE_CUDA
	md_free(data->linphase_gpu);
	md_free(data->psf_gpu);
	md_free(data->grid_gpu);
#endif
	linop_free(data->fft_op);
	if (data->conf.pcycle)
		linop_free(data->cfft_op);

	free(data);
}




// Forward: from image to kspace
static void nufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	struct nufft_data* data = CAST_DOWN(nufft_data, _data);

#ifdef USE_CUDA
	assert(!cuda_ondevice(src));
#endif
	assert(!data->conf.toeplitz); // if toeplitz linphase has no roll, so would need to be added

	unsigned int ND = data->N + 3;

	md_zmul2(ND, data->cml_dims, data->cml_strs, data->grid, data->cim_strs, src, data->lph_strs, data->linphase);
	linop_forward(data->fft_op, ND, data->cml_dims, data->grid, ND, data->cml_dims, data->grid);
	md_zmul2(ND, data->cml_dims, data->cml_strs, data->grid, data->cml_strs, data->grid, data->img_strs, data->fftmod);

	md_clear(ND, data->ksp_dims, dst, CFL_SIZE);

	complex float* gridX = md_alloc(data->N, data->cm2_dims, CFL_SIZE);

	long factors[data->N];

	for (unsigned int i = 0; i < data->N; i++)
		factors[i] = ((data->img_dims[i] > 1) && (i < 3)) ? 2 : 1;

	md_recompose(data->N, factors, data->cm2_dims, gridX, data->cml_dims, data->grid, CFL_SIZE);

	grid2H(2., data->width, data->beta, ND, data->trj_dims, data->traj, data->ksp_dims, dst, data->cm2_dims, gridX);

	md_free(gridX);

	if (NULL != data->weights)
		md_zmul2(data->N, data->ksp_dims, data->ksp_strs, dst, data->ksp_strs, dst, data->wgh_strs, data->weights);
}


// Adjoint: from kspace to image
static void nufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	struct nufft_data* data = CAST_DOWN(nufft_data, _data);

#ifdef USE_CUDA
	assert(!cuda_ondevice(src));
#endif
	unsigned int ND = data->N + 3;

	complex float* gridX = md_calloc(data->N, data->cm2_dims, CFL_SIZE);

	complex float* wdat = NULL;

	if (NULL != data->weights) {

		wdat = md_alloc(data->N, data->ksp_dims, CFL_SIZE);
		md_zmulc2(data->N, data->ksp_dims, data->ksp_strs, wdat, data->ksp_strs, src, data->wgh_strs, data->weights);
		src = wdat;
	}

	grid2(2., data->width, data->beta, ND, data->trj_dims, data->traj, data->cm2_dims, gridX, data->ksp_dims, src);

	md_free(wdat);

	long factors[data->N];

	for (unsigned int i = 0; i < data->N; i++)
		factors[i] = ((data->img_dims[i] > 1) && (i < 3)) ? 2 : 1;

	md_decompose(data->N, factors, data->cml_dims, data->grid, data->cm2_dims, gridX, CFL_SIZE);
	md_free(gridX);
	md_zmulc2(ND, data->cml_dims, data->cml_strs, data->grid, data->cml_strs, data->grid, data->img_strs, data->fftmod);
	linop_adjoint(data->fft_op, ND, data->cml_dims, data->grid, ND, data->cml_dims, data->grid);

	md_clear(ND, data->cim_dims, dst, CFL_SIZE);
	md_zfmacc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, data->grid, data->lph_strs, data->linphase);

	if (data->conf.toeplitz)
		md_zmul2(ND, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst, data->img_strs, data->roll);
}



/** 
 *
 */
static void nufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	struct nufft_data* data = CAST_DOWN(nufft_data, _data);

	if (data->conf.toeplitz) {

		if (data->conf.pcycle)
			toeplitz_mult_pcycle(data, dst, src);
		else
			toeplitz_mult(data, dst, src);

	} else {

		complex float* tmp_ksp = md_alloc(data->N + 3, data->ksp_dims, CFL_SIZE);

		nufft_apply(_data, tmp_ksp, src);
		nufft_apply_adjoint(_data, dst, tmp_ksp);

		md_free(tmp_ksp);
	}
}



static void toeplitz_mult(const struct nufft_data* data, complex float* dst, const complex float* src)
{
	unsigned int ND = data->N + 3;

	const complex float* linphase = data->linphase;
	const complex float* psf = data->psf;
	complex float* grid = data->grid;

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->linphase_gpu)
			((struct nufft_data*)data)->linphase_gpu = md_gpu_move(ND, data->lph_dims, data->linphase, CFL_SIZE);

		if (NULL == data->psf_gpu)
			((struct nufft_data*)data)->psf_gpu = md_gpu_move(ND, data->psf_dims, data->psf, CFL_SIZE);

		if (NULL == data->grid_gpu)
			((struct nufft_data*)data)->grid_gpu = md_gpu_move(ND, data->cml_dims, data->grid, CFL_SIZE);

		linphase = data->linphase_gpu;
		psf = data->psf_gpu;
		grid = data->grid_gpu;
	}
#endif
	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cim_strs, src, data->lph_strs, linphase);

	linop_forward(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);
	md_zmul2(ND, data->cml_dims, data->cml_strs, grid, data->cml_strs, grid, data->psf_strs, psf);
	linop_adjoint(data->fft_op, ND, data->cml_dims, grid, ND, data->cml_dims, grid);

	md_clear(ND, data->cim_dims, dst, CFL_SIZE);
	md_zfmacc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, grid, data->lph_strs, linphase);
}

static void toeplitz_mult_pcycle(const struct nufft_data* data, complex float* dst, const complex float* src)
{
	unsigned int ncycles = data->lph_dims[data->N];
        ((struct nufft_data*) data)->cycle = (data->cycle + 1) % ncycles;
	
	const complex float* clinphase = data->linphase + data->cycle * md_calc_size(data->N, data->lph_dims);
	const complex float* cpsf = data->psf + data->cycle * md_calc_size(data->N, data->psf_dims);
	complex float* grid = data->grid;

#ifdef USE_CUDA
	printf("Not Implemented.\n");
	exit(1);
#endif
	md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, src, data->img_strs, clinphase);

	linop_forward(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);
	md_zmul2(data->N, data->cim_dims, data->cim_strs, grid, data->cim_strs, grid, data->img_strs, cpsf);
	linop_adjoint(data->cfft_op, data->N, data->cim_dims, grid, data->N, data->cim_dims, grid);

	md_zmulc2(data->N, data->cim_dims, data->cim_strs, dst, data->cim_strs, grid, data->img_strs, clinphase);
}



/**
 * Estimate image dimensions from trajectory
 */
void estimate_im_dims(unsigned int N, long dims[3], const long tdims[N], const complex float* traj)
{
	float max_dims[3] = { 0., 0., 0. };

	for (long i = 0; i < md_calc_size(N - 1, tdims + 1); i++)
		for(int j = 0; j < 3; j++)
			max_dims[j] = MAX(cabsf(traj[j + tdims[0] * i]), max_dims[j]);

	for (int j = 0; j < 3; j++)
		dims[j] = (0. == max_dims[j]) ? 1 : (2 * (long)((2. * max_dims[j] + 1.5) / 2.));
}



