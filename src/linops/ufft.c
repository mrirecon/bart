/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "ufft.h"




/**
 * data structure for holding the undersampled fft data.
 */
struct ufft_data {

	INTERFACE(linop_data_t);

	bool use_gpu;
	unsigned int flags;

	long ksp_dims[DIMS];
	long pat_dims[DIMS];

	long ksp_strs[DIMS];
	long pat_strs[DIMS];

	const struct linop_s* fft_op;

        complex float* pat;
};


DEF_TYPEID(ufft_data);


static struct ufft_data* ufft_create_data(const long ksp_dims[DIMS], const long pat_dims[DIMS], const complex float* pat, unsigned int flags, bool use_gpu);
static void ufft_free_data(const linop_data_t* _data);
static void ufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src);
static void ufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src);
static void ufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src);
static void ufft_apply_pinverse(const linop_data_t* _data, float rho, complex float* dst, const complex float* src);



/**
 * Create undersampled/weighted fft operator
 */
const struct linop_s* linop_ufft_create(const long ksp_dims[DIMS], const long pat_dims[DIMS], const complex float* pat, unsigned int flags, bool use_gpu)
{
	struct ufft_data* data = ufft_create_data(ksp_dims, pat_dims, pat, flags, use_gpu);

	// Create operator interface
	return linop_create(DIMS, data->ksp_dims, DIMS, data->ksp_dims, CAST_UP(data),
		ufft_apply, ufft_apply_adjoint, ufft_apply_normal, ufft_apply_pinverse, ufft_free_data);
}


static struct ufft_data* ufft_create_data(const long ksp_dims[DIMS], const long pat_dims[DIMS], const complex float* pat, unsigned int flags, bool use_gpu)
{
	PTR_ALLOC(struct ufft_data, data);
	SET_TYPEID(ufft_data, data);

	data->flags = flags;
	data->use_gpu = use_gpu;

	md_copy_dims(DIMS, data->pat_dims, pat_dims);
	md_copy_dims(DIMS, data->ksp_dims, ksp_dims);

	md_calc_strides(DIMS, data->pat_strs, pat_dims, CFL_SIZE);
	md_calc_strides(DIMS, data->ksp_strs, ksp_dims, CFL_SIZE);

#ifdef USE_CUDA
	data->pat = (use_gpu ? md_alloc_gpu : md_alloc)(DIMS, data->pat_dims, CFL_SIZE);
#else
	data->pat = md_alloc(DIMS, data->pat_dims, CFL_SIZE);
#endif
	md_copy(DIMS, data->pat_dims, data->pat, pat, CFL_SIZE);

	data->fft_op = linop_fftc_create(DIMS, ksp_dims, flags);

	return PTR_PASS(data);
}


static void ufft_free_data(const linop_data_t* _data)
{
        struct ufft_data* data = CAST_DOWN(ufft_data, _data);

	md_free(data->pat);
	linop_free(data->fft_op);

	xfree(data);
}




/**
 * Undersampled FFT forward operator
 */
void ufft_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        struct ufft_data* data = CAST_DOWN(ufft_data, _data);

	linop_forward(data->fft_op, DIMS, data->ksp_dims, dst, DIMS, data->ksp_dims, src);

	md_zmul2(DIMS, data->ksp_dims, data->ksp_strs, dst, data->ksp_strs, dst, data->pat_strs, data->pat);
}


/**
 * Undersampled FFT adjoint operator
 */
void ufft_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        struct ufft_data* data = CAST_DOWN(ufft_data, _data);

	md_zmul2(DIMS, data->ksp_dims, data->ksp_strs, dst, data->ksp_strs, src, data->pat_strs, data->pat);

	linop_adjoint(data->fft_op, DIMS, data->ksp_dims, dst, DIMS, data->ksp_dims, dst);
}


/**
 * Undersampled FFT normal operator
 * X = pat^2 B
 */
void ufft_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        struct ufft_data* data = CAST_DOWN(ufft_data, _data);

	linop_forward(data->fft_op, DIMS, data->ksp_dims, dst, DIMS, data->ksp_dims, src);

	md_zmul2(DIMS, data->ksp_dims, data->ksp_strs, dst, data->ksp_strs, dst, data->pat_strs, data->pat);

	linop_adjoint(data->fft_op, DIMS, data->ksp_dims, dst, DIMS, data->ksp_dims, dst);
}


/**
 * 1/2 || Ax - b ||^2 + rho/2 || x - y ||^2
 * 
 * x = (ATA + lI)^-1 b
 *
 * X = 1 / (pat^2 + l) B
 *
 */
static void ufft_apply_pinverse(const linop_data_t* _data, float rho, complex float* dst, const complex float* src)
{
        struct ufft_data* data = CAST_DOWN(ufft_data, _data);

	md_zsadd(DIMS, data->pat_dims, data->pat, data->pat, rho);

	linop_forward(data->fft_op, DIMS, data->ksp_dims, dst, DIMS, data->ksp_dims, src);

	md_zdiv2(DIMS, data->ksp_dims, data->ksp_strs, dst, data->ksp_strs, dst, data->pat_strs, data->pat);
	
	linop_adjoint(data->fft_op, DIMS, data->ksp_dims, dst, DIMS, data->ksp_dims, dst);

	md_zsadd(DIMS, data->pat_dims, data->pat, data->pat, -rho);
}


