/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Frank Ong <frankong@berkeley.edu>
 * 2014-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * Strang G. A proposal for Toeplitz matrix calculations.
 * Journal Studies in Applied Math. 1986; 74:171-17. (FIXME)
 *
 */

#include <math.h>
#include <complex.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"
#include "num/fft.h"
#include "num/ops.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "noncart/nufft.h"
#include "noncart/nufft_priv.h"

#include "precond.h"



#define FFT_FLAGS (MD_BIT(0)|MD_BIT(1)|MD_BIT(2))



/**
 * Compute Strang's circulant preconditioner
 *
 * Strang's reconditioner is simply the cropped psf in the image domain
 *
 */
static struct multiplace_array_s* compute_precond(unsigned int N, const long* pre_dims, const long* pre_strs, const long* psf_dims, const long* psf_strs, const complex float* psf, const complex float* linphase)
{
	int ND = N + 1;
	unsigned long flags = FFT_FLAGS;

	complex float* pre = md_alloc(ND, pre_dims, CFL_SIZE);
	complex float* psft = md_alloc(ND, psf_dims, CFL_SIZE);

	// Transform psf to image domain
	ifftuc(ND, psf_dims, flags, psft, psf);

	// Compensate for linear phase to get cropped psf
	md_clear(ND, pre_dims, pre, CFL_SIZE);
	md_zfmacc2(ND, psf_dims, pre_strs, pre, psf_strs, psft, psf_strs, linphase);
	
        md_free(psft);
        
	// Transform to Fourier domain
	fftuc(N, pre_dims, flags, pre, pre);

	md_zabs(N, pre_dims, pre, pre);
	md_zsadd(N, pre_dims, pre, pre, 1e-3);
	
	return multiplace_move_F(ND, pre_dims, CFL_SIZE, pre);
}



/**
 * NUFFT precondition internal data structure
 */
struct nufft_precond_data {

	INTERFACE(operator_data_t);

	unsigned int N;
	struct multiplace_array_s* pre; ///< Preconditioner

	long* cim_dims; ///< Coil image dimension
	long* pre_dims; ///< Preconditioner dimension

	long* cim_strs;
	long* pre_strs;

	const struct linop_s* fft_op; ///< FFT linear operator
};


static DEF_TYPEID(nufft_precond_data);


static void nufft_precond_apply(const operator_data_t* _data, unsigned int M, void* args[M])
{
	assert(2 == M);

	const auto data = CAST_DOWN(nufft_precond_data, _data);

	complex float* dst = args[0];
	const complex float* src = args[1];

	linop_forward(data->fft_op, data->N, data->cim_dims, dst, data->N, data->cim_dims, src);

	md_zdiv2(data->N, data->cim_dims, data->cim_strs, dst, data->cim_strs, dst, data->pre_strs, multiplace_read(data->pre, dst));
	linop_adjoint(data->fft_op, data->N, data->cim_dims, dst, data->N, data->cim_dims, dst);
}

static void nufft_precond_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(nufft_precond_data, _data);

	xfree(data->cim_dims);
	xfree(data->pre_dims);
	xfree(data->cim_strs);
	xfree(data->pre_strs);

	multiplace_free(data->pre);
	xfree(data);
}

const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op)
{
	const auto data = CAST_DOWN(nufft_data, linop_get_data(nufft_op));

	PTR_ALLOC(struct nufft_precond_data, pdata);
	SET_TYPEID(nufft_precond_data, pdata);

	assert(data->conf.toeplitz);

	int N = data->N;
	int ND = N + 1;

	pdata->N = N;
	pdata->cim_dims = *TYPE_ALLOC(long[ND]);
	pdata->pre_dims = *TYPE_ALLOC(long[ND]);
	pdata->cim_strs = *TYPE_ALLOC(long[ND]);
	pdata->pre_strs = *TYPE_ALLOC(long[ND]);

	md_copy_dims(ND, pdata->cim_dims, data->cim_dims);
	md_select_dims(ND, data->flags, pdata->pre_dims, pdata->cim_dims);

	md_calc_strides(ND, pdata->cim_strs, pdata->cim_dims, CFL_SIZE);
	md_calc_strides(ND, pdata->pre_strs, pdata->pre_dims, CFL_SIZE);

	int cpu_ptr = 0;
	pdata->pre = compute_precond(pdata->N, pdata->pre_dims, pdata->pre_strs,
					data->psf_dims, data->psf_strs, multiplace_read(data->psf, &cpu_ptr),
					multiplace_read(data->linphase, &cpu_ptr));

	pdata->fft_op = linop_fft_create(pdata->N, pdata->cim_dims, data->flags);

	const long* cim_dims = pdata->cim_dims;	// need to dereference pdata before PTR_PASS

	return operator_create(N, cim_dims, N, cim_dims, CAST_UP(PTR_PASS(pdata)), nufft_precond_apply, nufft_precond_del);
}



