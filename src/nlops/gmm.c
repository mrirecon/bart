/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Tina Holliber
 */

#include <complex.h>
#include <stdio.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/gaussians.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/multiplace.h"

#include "nlops/nlop.h"

#include "gmm.h"


struct gmm_s {

	nlop_data_t super;

	int N;
	const long* score_dims;
	const long* mean_dims;
	const long* var_dims;
	const long* wgh_dims;

	struct multiplace_array_s* mean;
	struct multiplace_array_s* var;
	struct multiplace_array_s* wgh;
};

DEF_TYPEID(gmm_s);

static void gmm_score_fun(const nlop_data_t* _data, int D, complex float* args[D])
{
	assert(3 == D);

	const auto data = CAST_DOWN(gmm_s, _data);

	complex float* score = args[0];
	const complex float* x = args[1];
	complex float* noise_level = args[2]; // complex standard deviation

	const complex float* mean = multiplace_read(data->mean, score);
	const complex float* var = multiplace_read(data->var, score);
	const complex float* wgh = multiplace_read(data->wgh, score);

	complex float* end_var = md_alloc_sameplace(data->N, data->var_dims, CFL_SIZE, score);
	complex float* var_noise_level = md_alloc_sameplace(data->N, MD_SINGLETON_DIMS(data->N), CFL_SIZE, score);

	// calculate variance from standard deviation
	md_zspow(data->N, MD_SINGLETON_DIMS(data->N), var_noise_level, noise_level, 2.0f);

	// add noise scale to variance of complex gaussian mixture model
	md_zadd2(data->N, data->var_dims, MD_STRIDES(data->N, data->var_dims, CFL_SIZE), end_var, MD_STRIDES(data->N, MD_SINGLETON_DIMS(data->N), CFL_SIZE), var_noise_level, MD_STRIDES(data->N, data->var_dims, CFL_SIZE), var);

	// calc complex score
	md_gaussian_score(data->N, data->score_dims, score, data->score_dims, x, data->mean_dims, mean, data->var_dims, end_var, data->wgh_dims, wgh);

	md_free(end_var);
	md_free(var_noise_level);

}

static void gmm_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(gmm_s, _data);

	xfree(data->score_dims);
	xfree(data->mean_dims);
	xfree(data->var_dims);
	xfree(data->wgh_dims);

	multiplace_free(data->mean);
	multiplace_free(data->var);
	multiplace_free(data->wgh);

	xfree(data);
}


/**
 * Create operator with score (i.e. log of the grad) of a Gaussian Mixture Model (GMM) (two inputs with x and noise scale, one output as score)
 * @param N #dimensions
 * @param *dims dimensions
 * @param mean mean of each gaussian in the gaussian mixture model
 * @param var variance of each gaussian in the gaussian mixture model
 * @param wgh weighting of each gaussian in the gaussian mixture model
 */
struct nlop_s* nlop_gmm_score_create(int N, const long score_dims[N], const long mean_dims[N], const _Complex float* mean, const long var_dims[N], const _Complex float* var, const long wgh_dims[N], const _Complex float* wgh)
{
	PTR_ALLOC(struct gmm_s, data);
	SET_TYPEID(gmm_s, data);

	data->N = N;
	data->score_dims = ARR_CLONE(long[N], score_dims);
	data->mean_dims = ARR_CLONE(long[N], mean_dims);
	data->var_dims = ARR_CLONE(long[N], var_dims);
	data->wgh_dims = ARR_CLONE(long[N], wgh_dims);

	data->mean = multiplace_move(N, mean_dims, CFL_SIZE, mean);
	data->var = multiplace_move(N, var_dims, CFL_SIZE, var);
	data->wgh = multiplace_move(N, wgh_dims, CFL_SIZE, wgh);

	long odims[1][N];
	md_copy_dims(N, odims[0], score_dims); // Output dimensions

	long idims[2][N];
	md_copy_dims(N, idims[0], score_dims); // Input dimensions
	md_singleton_dims(N, idims[1]); // Singleton (noise_level)

	return nlop_generic_create(1, N, odims, 2, N, idims, CAST_UP(PTR_PASS(data)), gmm_score_fun, NULL, NULL, NULL,NULL, gmm_del);
}

