/* Copyright 2025-2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/
 
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "linops/sum.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/gmm.h"

#include "nn/nn.h"
#include "nn/weights.h"
#include "nn/ext_wrapper.h"

#include "networks/score.h"
#include "networks/cunet.h"

#include "priors.h"

#define DIMS 16


static const struct nlop_s* compute_score(const struct nlop_s *nlop, bool real_valued,
				const long img_dims[DIMS],
				const long msk_dims[DIMS], const complex float *msk)
{
	nlop = nlop_reshape_in_F(nlop, 0, DIMS, img_dims);
	nlop = nlop_reshape_out_F(nlop, 0, DIMS, img_dims);

	auto par = nlop_generic_domain(nlop, 1);

	if (1 < md_calc_size(par->N, par->dims))
		nlop = nlop_chain2_FF(nlop_from_linop_F(linop_repmat_create(par->N, par->dims, ~0UL)), 0, nlop, 1);

	if (real_valued)
		nlop = nlop_prepend_FF(nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), sqrtf(0.5))), nlop, 1);

	if (NULL != msk) {

		assert(md_check_equal_dims(DIMS, msk_dims, img_dims, md_nontriv_dims(DIMS, msk_dims)));

		const struct linop_s* lop_msk = linop_cdiag_create(DIMS, img_dims, md_nontriv_dims(DIMS, msk_dims), msk);

		nlop = nlop_append_FF(nlop, 0, nlop_from_linop_F(lop_msk));
	}

	nlop = nlop_expectation_to_score(nlop);

	nlop_unset_derivatives(nlop);

	return nlop;
}


const struct nlop_s* prior_cunet(const char* cunet_weights, struct nn_cunet_conf_s* cunet_conf,
				bool real_valued, const long msk_dims[DIMS], complex float* mask,
				long img_dims[DIMS])
{
	const struct nlop_s* nlop = NULL;

	nn_t cunet = cunet_bart_create(cunet_conf, DIMS, img_dims);
	
	cunet = nn_denoise_precond_edm(cunet, -1., -1., 0.5, false);

	nn_weights_t weights = load_nn_weights(cunet_weights);

	nlop = nn_get_nlop_wo_weights_F(cunet, weights, true);

	nn_weights_free(weights);

	return compute_score(nlop, real_valued, img_dims, msk_dims, mask);
}


const struct nlop_s* prior_graph(const char* graph, bool real_valued, bool gpu,
		const long msk_dims[DIMS], complex float* mask, long img_dims[DIMS])
{
	const struct nlop_s* nlop = NULL;

	long batchsize = img_dims[BATCH_DIM];

	// generates nlop from tf or pt graph
	int DO[1] = { 3 };
	int DI[2] = { 3, 1 };
	long idims1[3] = { img_dims[0], img_dims[1], batchsize };
	long idims2[1] = { batchsize };

	const char* key = NULL;

	nlop = nlop_external_graph_create(graph, 1, DO, (const long*[1]) { idims1 },
			2, DI, (const long*[2]) {idims1, idims2}, gpu, key);

	return compute_score(nlop, real_valued, img_dims, msk_dims, mask);
}



const struct nlop_s* prior_gmm(const long means_dims[DIMS], const complex float* means,
				const long weights_dims0[DIMS], const complex float *weights0,
				const long vars_dims0[DIMS], const complex float *vars0,
				long img_dims[DIMS], float *min_var)
{
	const struct nlop_s* nlop = NULL;

	img_dims[0] = means_dims[0];
	img_dims[1] = means_dims[1];
	img_dims[2] = means_dims[2];

	long weights_dims[DIMS];
	long vars_dims[DIMS];

	complex float* weights = NULL;

	// check if ws are given, otherwise use uniform weights over all mean peaks

	if (NULL == weights0) {

		md_select_dims(DIMS, ~md_nontriv_dims(DIMS, img_dims), weights_dims, means_dims);

		weights = md_alloc_sameplace(DIMS, weights_dims, CFL_SIZE, means);

		long num_gaussians = md_calc_size(DIMS, weights_dims);

		md_zfill(DIMS, weights_dims, weights, 1. / num_gaussians);

		debug_printf(DP_WARN, "No weighting specified. Uniform weigths are set.\n");

	} else {

		md_copy_dims(DIMS, weights_dims, weights_dims0);

		weights = md_alloc_sameplace(DIMS, weights_dims, CFL_SIZE, means);

		md_copy(DIMS, weights_dims, weights, weights0, CFL_SIZE);

		float wsum = md_zasum(DIMS, weights_dims, weights);

		md_zsmul(DIMS, weights_dims, weights, weights, 1. / wsum);
	}


	complex float* vars = NULL;

	if (NULL == vars0) {

		md_copy_dims(DIMS, vars_dims, weights_dims);

		vars = md_alloc_sameplace(DIMS, vars_dims, CFL_SIZE, means);

		md_zfill(DIMS, vars_dims, vars, 0.);

		debug_printf(DP_WARN, "No variance specified. Set to 0.\n");

	} else {

		md_copy_dims(DIMS, vars_dims, vars_dims0);

		vars = md_alloc_sameplace(DIMS, vars_dims, CFL_SIZE, means);

		md_copy(DIMS, vars_dims, vars, vars0, CFL_SIZE);
	}

	assert(md_check_equal_dims(DIMS, means_dims, vars_dims, ~md_nontriv_dims(DIMS, img_dims)));
	assert(md_check_equal_dims(DIMS, means_dims, weights_dims, ~md_nontriv_dims(DIMS, img_dims)));

	// Find minimum element in vars
	long num_elements = md_calc_size(DIMS, vars_dims);

	*min_var = crealf(vars[0]);

	for (long i = 1; i < num_elements; i++) {

		float v = crealf(vars[i]);

		if (v < *min_var)
			*min_var = v;
	}

	debug_printf(DP_DEBUG2, "Minimum variance in vars: %f\n", *min_var);

	nlop = nlop_gmm_score_create(DIMS, img_dims, means_dims, means, vars_dims, vars, weights_dims, weights);

	md_free(weights);
	md_free(vars);

	return nlop;
}
