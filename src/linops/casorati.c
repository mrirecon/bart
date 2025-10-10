/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <complex.h>

#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/casorati.h"
#include "num/fft.h"
#include "num/multiplace.h"
#include "num/linalg_rand.h"

#include "linops/linop.h"

#include "casorati.h"


struct casorati_s {

	linop_data_t super;

	unsigned long fflags;

	int N;
	const long* ddims;

	const long* odims;
	const long* idims;

	const long* wodims;
	const long* widims;

	const long* modims;
	const long* midims;

	struct multiplace_array_s* kern;
	struct multiplace_array_s* omask;
	struct multiplace_array_s* imask;
};

static DEF_TYPEID(casorati_s);

static struct casorati_s* casorati_data_create(int N, const long kdim[N], const long ddims[N], const complex float* data)
{
	PTR_ALLOC(struct casorati_s, d);
	SET_TYPEID(casorati_s, d);

	long odim[N];
	long idim[N];

	// batch dimensions (not allowed by explicit casorati)
	unsigned long bflags = ~md_nontriv_dims(N, ddims);

	for (int i = 0; i < N; i++) {

		assert(MD_IS_SET(bflags, i) || ddims[i] >= kdim[i]);

		odim[i] = MD_IS_SET(bflags, i) ? kdim[i] : ddims[i] - kdim[i] + 1;	// number of shifted blocks
		idim[i] = kdim[i];							// size of blocks
	}

	unsigned long tflags = ~md_nontriv_dims(N, idim) | ~md_nontriv_dims(N, odim);
	unsigned long fflags = ~bflags & ~tflags;

	long widims[N];
	long wodims[N];
	md_select_dims(N, md_nontriv_dims(N, idim), widims, ddims);
	md_select_dims(N, md_nontriv_dims(N, odim), wodims, ddims);
	md_max_dims(N, bflags, widims, widims, idim);
	md_max_dims(N, bflags, wodims, wodims, odim);

	complex float* kern = md_alloc_sameplace(N, ddims, CFL_SIZE, data);
	ifft(N, ddims, fflags, kern, data);
	long fft_dims[N];
	md_select_dims(N, fflags, fft_dims, ddims);
	md_zsmul(N, ddims, kern, kern, 1. / md_calc_size(N, fft_dims));
	d->kern = multiplace_move_F(N, ddims, CFL_SIZE, kern);

	long modims[N];
	long midims[N];
	long tdims[N];
	md_select_dims(N, fflags, modims, wodims);
	md_select_dims(N, fflags, midims, widims);

	complex float* omask = md_alloc_sameplace(N, modims, CFL_SIZE, data);
	md_clear(N, modims, omask, CFL_SIZE);
	md_select_dims(N, fflags, tdims, odim);
	md_zfill2(N, tdims, MD_STRIDES(N, modims, CFL_SIZE), omask, 1.);
	d->omask = multiplace_move_F(N, modims, CFL_SIZE, omask);

	complex float* imask = md_alloc_sameplace(N, midims, CFL_SIZE, data);
	md_clear(N, midims, imask, CFL_SIZE);
	md_select_dims(N, fflags, tdims, idim);
	md_zfill2(N, tdims, MD_STRIDES(N, midims, CFL_SIZE), imask, 1.);
	d->imask = multiplace_move_F(N, midims, CFL_SIZE, imask);

	d->N = N;
	d->ddims = ARR_CLONE(long[N], ddims);
	d->odims = ARR_CLONE(long[N], odim);
	d->idims = ARR_CLONE(long[N], idim);
	d->wodims = ARR_CLONE(long[N], wodims);
	d->widims = ARR_CLONE(long[N], widims);
	d->modims = ARR_CLONE(long[N], modims);
	d->midims = ARR_CLONE(long[N], midims);

	d->fflags = fflags;

	return PTR_PASS(d);
}

static void casorati_data_free(const linop_data_t* _data)
{
	auto data = CAST_DOWN(casorati_s, _data);

	xfree(data->ddims);
	xfree(data->odims);
	xfree(data->idims);
	xfree(data->wodims);
	xfree(data->widims);
	xfree(data->modims);
	xfree(data->midims);

	multiplace_free(data->kern);
	multiplace_free(data->omask);
	multiplace_free(data->imask);

	xfree(data);
}

static void casorati_forward(const linop_data_t* data, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(casorati_s, data);

	complex float* worki = md_alloc_sameplace(d->N, d->widims, CFL_SIZE, src);
	md_clear(d->N, d->widims, worki, CFL_SIZE);

	md_copy2(d->N, d->idims, MD_STRIDES(d->N, d->widims, CFL_SIZE), worki, MD_STRIDES(d->N, d->idims, CFL_SIZE), src, CFL_SIZE);

	fft(d->N, d->widims, d->fflags, worki, worki);

	complex float* worko = md_alloc_sameplace(d->N, d->wodims, CFL_SIZE, src);
	md_ztenmul(d->N, d->wodims, worko, d->widims, worki, d->ddims, multiplace_read(d->kern, src));
	md_free(worki);

	fft(d->N, d->wodims, d->fflags, worko, worko);

	md_copy2(d->N, d->odims, MD_STRIDES(d->N, d->odims, CFL_SIZE), dst, MD_STRIDES(d->N, d->wodims, CFL_SIZE), worko, CFL_SIZE);

	md_free(worko);
}



static void casorati_normal(const linop_data_t* data, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(casorati_s, data);

	complex float* worki = md_alloc_sameplace(d->N, d->widims, CFL_SIZE, src);
	complex float* worko = md_alloc_sameplace(d->N, d->wodims, CFL_SIZE, src);

	md_clear(d->N, d->widims, worki, CFL_SIZE);
	md_copy2(d->N, d->idims, MD_STRIDES(d->N, d->widims, CFL_SIZE), worki, MD_STRIDES(d->N, d->idims, CFL_SIZE), src, CFL_SIZE);

	fft(d->N, d->widims, d->fflags, worki, worki);

	md_ztenmul(d->N, d->wodims, worko, d->widims, worki, d->ddims, multiplace_read(d->kern, src));

	fft(d->N, d->wodims, d->fflags, worko, worko);

	md_zmul2(d->N, d->wodims, MD_STRIDES(d->N, d->wodims, CFL_SIZE), worko, MD_STRIDES(d->N, d->wodims, CFL_SIZE), worko, MD_STRIDES(d->N, d->modims, CFL_SIZE), multiplace_read(d->omask, src));

	ifft(d->N, d->wodims, d->fflags, worko, worko);

	md_ztenmulc(d->N, d->widims, worki, d->wodims, worko, d->ddims, multiplace_read(d->kern, src));

	ifft(d->N, d->widims, d->fflags, worki, worki);

	md_copy2(d->N, d->idims, MD_STRIDES(d->N, d->idims, CFL_SIZE), dst, MD_STRIDES(d->N, d->widims, CFL_SIZE), worki, CFL_SIZE);

	md_free(worki);
	md_free(worko);
}

static void casorati_adjoint(const linop_data_t* data, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(casorati_s, data);

	complex float* worko = md_alloc_sameplace(d->N, d->wodims, CFL_SIZE, src);

	md_clear(d->N, d->wodims, worko, CFL_SIZE);
	md_copy2(d->N, d->odims, MD_STRIDES(d->N, d->wodims, CFL_SIZE), worko, MD_STRIDES(d->N, d->odims, CFL_SIZE), src, CFL_SIZE);

	ifft(d->N, d->wodims, d->fflags, worko, worko);

	complex float* worki = md_alloc_sameplace(d->N, d->widims, CFL_SIZE, src);
	md_ztenmulc(d->N, d->widims, worki, d->wodims, worko, d->ddims, multiplace_read(d->kern, src));
	md_free(worko);

	ifft(d->N, d->widims, d->fflags, worki, worki);

	md_copy2(d->N, d->idims, MD_STRIDES(d->N, d->idims, CFL_SIZE), dst, MD_STRIDES(d->N, d->widims, CFL_SIZE), worki, CFL_SIZE);

	md_free(worki);
}

static void casoratiH_normal(const linop_data_t* data, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(casorati_s, data);

	complex float* worki = md_alloc_sameplace(d->N, d->widims, CFL_SIZE, src);
	complex float* worko = md_alloc_sameplace(d->N, d->wodims, CFL_SIZE, src);

	md_clear(d->N, d->wodims, worko, CFL_SIZE);
	md_copy2(d->N, d->odims, MD_STRIDES(d->N, d->wodims, CFL_SIZE), worko, MD_STRIDES(d->N, d->odims, CFL_SIZE), src, CFL_SIZE);

	ifft(d->N, d->wodims, d->fflags, worko, worko);

	md_ztenmulc(d->N, d->widims, worki, d->wodims, worko, d->ddims, multiplace_read(d->kern, src));

	ifft(d->N, d->widims, d->fflags, worki, worki);

	md_zmul2(d->N, d->widims, MD_STRIDES(d->N, d->widims, CFL_SIZE), worki, MD_STRIDES(d->N, d->widims, CFL_SIZE), worki, MD_STRIDES(d->N, d->midims, CFL_SIZE), multiplace_read(d->imask, src));

	fft(d->N, d->widims, d->fflags, worki, worki);

	md_ztenmul(d->N, d->wodims, worko, d->widims, worki, d->ddims, multiplace_read(d->kern, src));

	fft(d->N, d->wodims, d->fflags, worko, worko);

	md_copy2(d->N, d->odims, MD_STRIDES(d->N, d->odims, CFL_SIZE), dst, MD_STRIDES(d->N, d->wodims, CFL_SIZE), worko, CFL_SIZE);

	md_free(worki);
	md_free(worko);
}

/**
 * Create linop applying the casorati matrix on its input
 * @param N number of dimensions
 * @param kdim dimensions of blocks
 * @param ddims dimensions of data
 * @param data casorati kernel (in image space)
 * @return linop
 * if ddims is singleton, kdim can be non-singleton which is interpreted as a batch dimension
*/
const struct linop_s* linop_casorati_create(int N, const long kdim[N], const long ddims[N], const complex float* data)
{
	auto d = casorati_data_create(N, kdim, ddims, data);

	return linop_create(N, d->odims, N, d->idims, CAST_UP(d), casorati_forward, casorati_adjoint, casorati_normal, NULL, casorati_data_free);
}

/**
 * Create linop applying the adjoint casorati matrix on its input
 * @param N number of dimensions
 * @param kdim dimensions of blocks
 * @param ddims dimensions of data
 * @param data casorati kernel (in image space)
 * @return linop
 * if ddims is singleton, kdim can be non-singleton which is interpreted as a batch dimension
*/
const struct linop_s* linop_casoratiH_create(int N, const long kdim[N], const long ddims[N], const complex float* data)
{
	auto d = casorati_data_create(N, kdim, ddims, data);

	return linop_create(N, d->idims, N, d->odims, CAST_UP(d), casorati_adjoint, casorati_forward, casoratiH_normal, NULL, casorati_data_free);
}


void casorati_gram(int M, complex float out[M][M], int N, const long kdims[N], const long dims[N], const complex float* data)
{
	assert(M == md_calc_size(N, kdims));

	long kdimsB[N + 1];
	long dimsB[N + 1];

	kdimsB[N] = M;
	dimsB[N] = 1;
	md_copy_dims(N, kdimsB, kdims);
	md_copy_dims(N, dimsB, dims);

	const struct linop_s* lop_casorati = linop_casorati_create(N + 1, kdimsB, dimsB, data);

	long odims[2] = { M, M };

	complex float* id = md_alloc_sameplace(2, odims, CFL_SIZE, &out[0][0]);
	md_clear(2, odims, id, CFL_SIZE);

	complex float* one = md_alloc_sameplace(1, MD_DIMS(1), CFL_SIZE, &out[0][0]);
	md_zfill(1, MD_DIMS(1), one, 1.);
	md_copy2(1, MD_DIMS(M), MD_DIMS((1 + M) * (long)CFL_SIZE), id, MD_DIMS(0), one, CFL_SIZE);
	md_free(one);

	linop_normal_unchecked(lop_casorati, &out[0][0], id);

	linop_free(lop_casorati);
	md_free(id);

	//FIXME: C vs Fortran order but cheaper
	//	 transpose is equivalent to conjugate for self-adjoint matrix
	md_zconj(2, MD_DIMS(M, M), &out[0][0], &out[0][0]);
}


void casorati_gram_eig_nystroem(int K, int P, int M, float eig[K], complex float out[K][M], int N, const long kdims[N], const long dims[N], const complex float* data)
{
	assert(M == md_calc_size(N, kdims));

	long kdimsB[N + 1];
	long dimsB[N + 1];

	kdimsB[N] = K + P;
	dimsB[N] = 1;
	md_copy_dims(N, kdimsB, kdims);
	md_copy_dims(N, dimsB, dims);

	const struct linop_s* lop_casorati = linop_casorati_create(N + 1, kdimsB, dimsB, data);
	randomized_eig_block(lop_casorati->normal, 1, M, K, P, out, eig);
	linop_free(lop_casorati);

	md_zconj(2, MD_DIMS(M, K), &out[0][0], &out[0][0]);
}



