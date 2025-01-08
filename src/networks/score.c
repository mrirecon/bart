/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <math.h>

#include "iter/italgos.h"

#include "num/multind.h"
#include "num/iovec.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/cast.h"
#include "nlops/zexp.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"
#include "nlops/mri_ops.h"
#include "nlops/losses.h"

#include "nn/nn.h"
#include "nn/chain.h"
#include "nn/losses.h"

#include "score.h"


//D(x, s) = s^2 * score(x, s) + x
nn_t nn_score_to_expectation(nn_t score)
{
	auto dom = nn_generic_domain(score, 0, NULL);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	auto sdom = nn_generic_domain(score, 1, NULL);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	const struct nlop_s* nlop_scale = nlop_tenmul_create(N, dims, dims, sdims);
	nlop_scale = nlop_prepend_FF(nlop_zspow_create(N, sdims, 2), nlop_scale, 1);
	nlop_scale = nlop_reshape_in_F(nlop_scale, 1, sdom->N, sdom->dims);

	const struct nlop_s* nlop_skip = nlop_zaxpbz_create(N, dims, 1, 1);

	auto ret = nn_chain2_swap_FF(score, 0, NULL, nn_from_nlop_F(nlop_scale), 0, NULL);
	ret = nn_dup_F(ret, 1, NULL, 2, NULL);

	ret = nn_chain2_FF(ret, 0, NULL, nn_from_nlop_F(nlop_skip), 0, NULL);
	ret = nn_dup_F(ret, 0, NULL, 1, NULL);

	return ret;
}

//score = (D(x, s) - x) / s^2
nn_t nn_expectation_to_score(nn_t Dx)
{
	auto dom = nn_generic_domain(Dx, 0, NULL);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	auto sdom = nn_generic_domain(Dx, 1, NULL);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	const struct nlop_s* nlop_scale = nlop_tenmul_create(N, dims, dims, sdims);
	nlop_scale = nlop_prepend_FF(nlop_zspow_create(N, sdims, -2), nlop_scale, 1);
	nlop_scale = nlop_reshape_in_F(nlop_scale, 1, sdom->N, sdom->dims);

	const struct nlop_s* nlop_skip = nlop_zaxpbz_create(N, dims, -1, 1);

	auto ret = nn_chain2_FF(Dx, 0, NULL, nn_from_nlop_F(nlop_skip), 1, NULL);
	ret = nn_dup_F(ret, 0, NULL, 1, NULL);

	ret = nn_chain2_swap_FF(ret, 0, NULL, nn_from_nlop_F(nlop_scale), 0, NULL);
	ret = nn_dup_F(ret, 1, NULL, 2, NULL);

	return ret;
}


//D(x, s) = s^2 * score(x, s) + x
const struct nlop_s* nlop_score_to_expectation(const struct nlop_s* score)
{
	auto dom = nlop_generic_domain(score, 0);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	auto sdom = nlop_generic_domain(score, 1);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	const struct nlop_s* nlop_scale = nlop_tenmul_create(N, dims, dims, sdims);
	nlop_scale = nlop_prepend_FF(nlop_zspow_create(N, sdims, 2), nlop_scale, 1);
	nlop_scale = nlop_reshape_in_F(nlop_scale, 1, sdom->N, sdom->dims);

	const struct nlop_s* nlop_skip = nlop_zaxpbz_create(N, dims, 1, 1);

	auto ret = nlop_chain2_swap_FF(score, 0, nlop_scale, 0);
	ret = nlop_dup_F(ret, 1, 2);

	ret = nlop_chain2_FF(ret, 0, nlop_skip, 0);
	ret = nlop_dup_F(ret, 0, 1);

	return ret;
}

//score = (D(x, s) - x) / s^2
const struct nlop_s* nlop_expectation_to_score(const struct nlop_s* Dx)
{
	auto dom = nlop_generic_domain(Dx, 0);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	auto sdom = nlop_generic_domain(Dx, 1);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	const struct nlop_s* nlop_scale = nlop_tenmul_create(N, dims, dims, sdims);
	nlop_scale = nlop_prepend_FF(nlop_zspow_create(N, sdims, -2), nlop_scale, 1);
	nlop_scale = nlop_reshape_in_F(nlop_scale, 1, sdom->N, sdom->dims);

	const struct nlop_s* nlop_skip = nlop_zaxpbz_create(N, dims, -1, 1);

	auto ret = nlop_chain2_FF(Dx, 0, nlop_skip, 1);
	ret = nlop_dup_F(ret, 0, 1);

	ret = nlop_chain2_swap_FF(ret, 0, nlop_scale, 0);
	ret = nlop_dup_F(ret, 1, 2);

	return ret;
}


// flag ambient is used for modified EDM, output of net is scaled with constant
extern const struct nn_s* nn_denoise_precond_edm(const struct nn_s* network, float /*sigma_min*/, float /*sigma_max*/, float sigma_data, _Bool ambient)
{
	auto dom = nn_generic_domain(network, 0, NULL);
	dom = iovec_create2(dom->N, dom->dims, dom->strs, dom->size);

	auto sdom = nn_generic_domain(network, 1, NULL);
	sdom = iovec_create2(sdom->N, sdom->dims, sdom->strs, sdom->size);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	network = nn_reshape_in_F(network, 1, NULL, N, sdims);

	const struct nlop_s* c_noise = nlop_zlog_create(N, sdims);  // in: sigma; out: log(sigma)
	c_noise = nlop_chain_FF(c_noise, nlop_from_linop_F(linop_scale_create(N, sdims, 0.25)));  // in: sigma; out: log(sigma) / 4
	network = nn_chain2_FF(nn_from_nlop_F(c_noise), 0, NULL, network, 1, NULL);  // in: (y+z*sigma), sigma; out: D_yn(y+z*sigma,sigma_noise)

	const struct nlop_s* c_in = nlop_zspow_create(N, sdims, 2.);  // in: sigma; out: sigma**2
	c_in = nlop_chain_FF(c_in, nlop_set_input_scalar_F(nlop_zaxpbz_create(N, sdims, 1, 1), 1, sigma_data * sigma_data));  // in: sigma; out: sigma**2 + sigma_data**2
	c_in = nlop_chain_FF(c_in, nlop_zspow_create(N, sdims, -0.5));  // in: sigma; out: 1 / sqrt(sigma**2 + sigma_data**2)
	c_in = nlop_prepend_FF(c_in, nlop_tenmul_create(N, dims, dims, sdims), 1);
	network = nn_chain2_swap_FF(nn_from_nlop_F(c_in), 0, NULL, network, 0, NULL);  // in: (y+z*sigma), sigma, sigma; out: D_yn(c_in*(y+z*sigma),sigma_noise)
	network = nn_dup_F(network, 1, NULL, 2, NULL);  // in: (y+z*sigma), sigma

	const struct nlop_s* c_out;
	if (ambient) {
		c_out = nlop_from_linop_F(linop_scale_create(N, dims, 1 / sigma_data));  // out: 1 / sigma_data;
		network = nn_chain2_swap_FF(network, 0, NULL, nn_from_nlop_F(c_out), 0, NULL);  // in: (y+z*sigma), sigma; out: c_out * D_yn(c_in*(y+z*sigma),sigma_noise)
	} else {

		c_out = nlop_zspow_create(N, sdims, 2.);  // in: sigma; out: sigma**2
		c_out = nlop_chain_FF(c_out, nlop_set_input_scalar_F(nlop_zaxpbz_create(N, sdims, 1, 1), 1, sigma_data * sigma_data));  // in: sigma; out: sigma**2 + sigma_data**2
		c_out = nlop_chain_FF(c_out, nlop_zspow_create(N, sdims, -0.5));  // in: sigma; out: 1 / sqrt(sigma**2 + sigma_data**2)
		c_out = nlop_chain2_FF(c_out, 0, nlop_tenmul_create(N, sdims, sdims, sdims), 0);  // in: sigma, sigma; out: sigma / sqrt(sigma**2 + sigma_data**2)
		c_out = nlop_dup_F(c_out, 0, 1);  // in: sigma
		c_out = nlop_chain_FF(c_out, nlop_from_linop_F(linop_scale_create(N, sdims, sigma_data)));  // in: sigma; out: sigma / sqrt(sigma**2 + sigma_data**2) * sigma_data
		c_out = nlop_chain2_FF(c_out, 0, nlop_tenmul_create(N, dims, dims, sdims), 1);  // in: x, sigma; out: sigma / sqrt(sigma**2 + sigma_data**2) * x
		network = nn_chain2_swap_FF(network, 0, NULL, nn_from_nlop_F(c_out), 0, NULL);  // in: (y+z*sigma), sigma, sigma; out: c_out * D_yn(c_in*(y+z*sigma),sigma_noise)
		network = nn_dup_F(network, 1, NULL, 2, NULL);  // in: (y+z*sigma), sigma
	}

	const struct nlop_s* c_skip = nlop_zspow_create(N, sdims, 2.);  // in: sigma; out: sigma**2
	c_skip = nlop_chain_FF(c_skip, nlop_set_input_scalar_F(nlop_zaxpbz_create(N, sdims, 1, 1), 1, sigma_data * sigma_data));  // in: sigma; out: sigma**2 + sigma_data**2
	c_skip = nlop_chain_FF(c_skip, nlop_zinv_create(N, sdims));  // in: sigma; out: 1 / (sigma**2 + sigma_data**2)
	c_skip = nlop_chain_FF(c_skip, nlop_from_linop_F(linop_scale_create(N, sdims, sigma_data * sigma_data)));  // in: sigma; out: sigma_data**2 / (sigma**2 + sigma_data**2)

	c_skip = nlop_prepend_FF(c_skip, nlop_tenmul_create(N, dims, dims, sdims), 1);
	c_skip = nlop_chain2_FF(c_skip, 0, nlop_zaxpbz_create(N, dims, 1., 1.), 0);  // in: (y+z*sigma), sigma; out: c_skip * (y+z*sigma)
	network = nn_chain2_swap_FF(network, 0, NULL, nn_from_nlop_F(c_skip), 0, NULL);  // in: (y+z*sigma), sigma, (y+z*sigma), sigma; out: c_out * D_yn(c_in*(y+z*sigma),sigma_noise) + c_skip * (y+z*sigma)
	network = nn_dup_F(network, 0, NULL, 2, NULL);  // in: (y+z*sigma), sigma, sigma
	network = nn_dup_F(network, 1, NULL, 2, NULL);  // in: (y+z*sigma), sigma
	network = nn_reshape_in_F(network, 1, NULL, sdom->N, sdom->dims);

	iovec_free(sdom);
	iovec_free(dom);

	return network;
}


extern const struct nn_s* nn_denoise_loss_VE(const struct nn_s* network, float sigma_min, float sigma_max, float /*sigma_data*/)
{
	auto dom = nn_generic_domain(network, 0, NULL);
	auto sdom = nn_generic_domain(network, 1, NULL);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	network = nn_reshape_in_F(network, 1, NULL, N, sdims);

	network = nn_chain2_swap_FF(nn_from_nlop_F(nlop_zaxpbz_create(N, dims, 1, 1)), 0, NULL, network, 0, NULL);  // in: y, sigma*z, sigma; out: D_yn(y+sigma*z; sigma)
	network = nn_chain2_swap_FF(nn_from_nlop_F(nlop_tenmul_create(N, dims, dims, sdims)), 0, NULL, network, 1, NULL);  // in: z, sigma, y, sigma; out: D_yn(y+sigma*z; sigma)
	network = nn_dup_F(network, 1, NULL, 3, NULL);  // in: z, sigma, y; out D_yn(y+sigma*z; sigma)
	network = nn_chain2_FF(network, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(N, dims, -1, 1)), 1, NULL);  // in: y, z, sigma; out: D_yn(y+sigma*z; sigma) - y
	network = nn_dup_F(network, 0, NULL, 3, NULL);

	const struct nlop_s* weight = nlop_zspow_create(N, sdims, -1.);
	weight = nlop_prepend_FF(weight, nlop_tenmul_create(N, dims, dims, sdims), 1);

	network = nn_chain2_swap_FF(network, 0, NULL, nn_from_nlop_F(weight), 0, NULL);  // in: y, z, sigma, sigma; out: 1/sigma (D_yn(y+sigma*z; sigma) - y)
	network = nn_dup_F(network, 2, NULL, 3, NULL);  // in: y, z, sigma; out: 1/sigma (D_yn(y+sigma*z; sigma) - y)
	network = nn_chain2_FF(network, 0, NULL, nn_from_nlop_F(nlop_znorm_create(N, dims, ~0UL)), 0, NULL);  // in: y, z, sigma; out: 1/N ||1/sigma (D_yn(y+sigma*z; sigma) - y)||^2

	const struct nlop_s* sigma = nlop_from_linop_F(linop_zreal_create(N, sdims));
	sigma = nlop_chain_FF(sigma, nlop_from_linop_F(linop_scale_create(N, sdims, logf(sigma_max / sigma_min))));
	sigma = nlop_chain_FF(sigma, nlop_zexp_create(N, sdims));
	sigma = nlop_chain_FF(sigma, nlop_from_linop_F(linop_scale_create(N, sdims, sigma_min)));

	network = nn_chain2_FF(nn_from_nlop_F(sigma), 0, NULL, network, 2, NULL);

	network = nn_set_in_type_F(network, 0, NULL, IN_BATCH_GENERATOR);
	network = nn_set_in_type_F(network, 1, NULL, IN_GAUSSIAN_RAND);
	network = nn_set_in_type_F(network, 2, NULL, IN_UNIFORM_RAND);

	network = nn_set_out_type_F(network, 0, NULL, OUT_OPTIMIZE);

	return network;
}


extern const struct nn_s* nn_denoise_loss_EDM(const struct nn_s* network, float /*sigma_min*/, float /*sigma_max*/, float sigma_data)
{
	auto dom = nn_generic_domain(network, 0, NULL);
	auto sdom = nn_generic_domain(network, 1, NULL);

	int N = dom->N;
	long dims[N];
	md_copy_dims(N, dims, dom->dims);

	long sdims[N];
	md_singleton_dims(N, sdims);
	sdims[N - 1] = md_calc_size(sdom->N, sdom->dims);

	network = nn_reshape_in_F(network, 1, NULL, N, sdims);

	network = nn_chain2_swap_FF(nn_from_nlop_F(nlop_zaxpbz_create(N, dims, 1, 1)), 0, NULL, network, 0, NULL);  // in: y, sigma*z, sigma; out: D_yn(y+sigma*z; sigma)
	network = nn_chain2_swap_FF(nn_from_nlop_F(nlop_tenmul_create(N, dims, dims, sdims)), 0, NULL, network, 1, NULL);  // in: z, sigma, y, sigma; out: D_yn(y+sigma*z; sigma)
	network = nn_dup_F(network, 1, NULL, 3, NULL);  // in: z, sigma, y; out D_yn(y+sigma*z; sigma)
	network = nn_chain2_FF(network, 0, NULL, nn_from_nlop_F(nlop_zaxpbz_create(N, dims, -1, 1)), 1, NULL);  // in: y, z, sigma; out: D_yn(y+sigma*z; sigma) - y
	network = nn_dup_F(network, 0, NULL, 3, NULL);

	const struct nlop_s* weight = nlop_zspow_create(N, sdims, 2.);
	weight = nlop_chain_FF(weight, nlop_set_input_scalar_F(nlop_zaxpbz_create(N, sdims, 1, 1), 1, sigma_data * sigma_data));
	weight = nlop_chain_FF(weight, nlop_zspow_create(N, sdims, -0.5));
	weight = nlop_chain2_FF(weight, 0, nlop_tenmul_create(N, sdims, sdims, sdims), 0);
	weight = nlop_dup_F(weight, 0, 1);
	weight = nlop_chain_FF(weight, nlop_from_linop_F(linop_scale_create(N, sdims, sigma_data)));
	weight = nlop_chain_FF(weight, nlop_zspow_create(N, sdims, -1.));  // in: sigma, out: sqrt(sigma^2 + sigma_d^2) / (sigma * sigma_d)
	weight = nlop_prepend_FF(weight, nlop_tenmul_create(N, dims, dims, sdims), 1);

	network = nn_chain2_swap_FF(network, 0, NULL, nn_from_nlop_F(weight), 0, NULL);  // in: y, z, sigma, sigma; out: 1/sigma (D_yn(y+sigma*z; sigma) - y)
	network = nn_dup_F(network, 2, NULL, 3, NULL);  // in: y, z, sigma; out: 1/sigma (D_yn(y+sigma*z; sigma) - y)
	network = nn_chain2_FF(network, 0, NULL, nn_from_nlop_F(nlop_znorm_create(N, dims, ~0UL)), 0, NULL);  // in: y, z, sigma; out: 1/N ||1/sigma (D_yn(y+sigma*z; sigma) - y)||^2

	float Pmean = -1.2;
	float Pstd = 1.2;

	const struct nlop_s* sigma = nlop_from_linop_F(linop_zreal_create(N, sdims));
	sigma = nlop_prepend_FF(sigma, nlop_zaxpbz_create(N, sdims, Pstd, 1.), 0);
	sigma = nlop_set_input_scalar_F(sigma, 1, Pmean);
	sigma = nlop_chain_FF(sigma, nlop_zexp_create(N, sdims));

	network = nn_chain2_FF(nn_from_nlop_F(sigma), 0, NULL, network, 2, NULL);

	network = nn_set_in_type_F(network, 0, NULL, IN_BATCH_GENERATOR);
	network = nn_set_in_type_F(network, 1, NULL, IN_GAUSSIAN_RAND);
	network = nn_set_in_type_F(network, 2, NULL, IN_GAUSSIAN_RAND);

	network = nn_set_out_type_F(network, 0, NULL, OUT_OPTIMIZE);

	return network;
}


