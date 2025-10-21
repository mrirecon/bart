/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *          Tina Holliber
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <string.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/init.h"
#include "num/iovec.h"
#include "num/rand.h"
#include "num/ops_p.h"

#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mri.h"

#include "linops/linop.h"
#include "linops/sum.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/gmm.h"

#include "nn/nn.h"
#include "nn/weights.h"
#include "nn/ext_wrapper.h"

#include "networks/score.h"

#include "iter/iter2.h"
#include "iter/iter.h"
#include "iter/prox2.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] =
	"Prior sampling with given diffusion network (either PyTorch or TensorFlow) which is trained as denoiser (i.e. outputs the expectation) or Gaussian Mixture Model using unadjusted Langevin algorithm.\n";


static void print_stats(int dl, float t, long img_dims[DIMS], const complex float* samples, float sigma)
{
	complex float* std_device = md_alloc_sameplace(1, MD_DIMS(1), CFL_SIZE, samples);

	md_zstd(DIMS, img_dims, ~0UL, std_device, samples);

	float std_samples;
	md_copy(1, MD_DIMS(1), &std_samples, std_device, FL_SIZE);

	long corn_dims[DIMS];
	md_copy_dims(DIMS, corn_dims, img_dims);
	corn_dims[0] = MIN(16, img_dims[0]);
	corn_dims[1] = MIN(16, img_dims[1]);

	md_zstd2(DIMS, corn_dims, ~0ul, MD_SINGLETON_STRS(DIMS), std_device, MD_STRIDES(DIMS, img_dims, CFL_SIZE), samples);

	float std_corner;
	md_copy(1, MD_DIMS(1), &std_corner, std_device, FL_SIZE);

	md_free(std_device);

	debug_printf(dl, "t=%.2f; sig=%.4f; zstd/sig=%.2f; zstd(corner)/sig=%.2f\n", t, sigma, std_samples / sigma, std_corner / sigma);
}


enum SIGMA_SCHEDULE { SIGMA_SCHEDULE_EXP, SIGMA_SCHEDULE_QUAD };

typedef float (*sigma_schedule_fun_t)(float t, float sigma_min, float sigma_max);

static float sigma_schedule_exp(float t, float sigma_min, float sigma_max)
{
	return sigma_min * expf(logf(sigma_max / sigma_min) * t);
}

static float sigma_schedule_quad(float t, float sigma_min, float sigma_max)
{
	return sigma_min + sigma_max * t * t;
}

int main_sample(int argc, char* argv[argc])
{
	const char* graph = NULL;
	const char* key = NULL;

	const char* means_file = NULL;
	const char* vars_file = NULL;
	const char* ws_file = NULL;

	const char* samples_file = NULL;
	const char* mmse_file = NULL;
	long batchsize = 1;
	unsigned int seed = 123;

	float sigma_min = 0.01;
	float sigma_max = 10.;
	enum SIGMA_SCHEDULE schedule = SIGMA_SCHEDULE_EXP;

	int N = 100;
	int K = 1;
	bool ancestral = false;
	bool predictor_corrector = false;
	bool real_valued = false;

	long save_mod = 0;

	float gamma_base = 0.5;

	long img_dims[DIMS];
	md_singleton_dims(DIMS, img_dims);
	img_dims[0] = 256;
	img_dims[1] = 256;

	struct arg_s args[] = {
		ARG_INOUTFILE(true, &samples_file, "samples"),
		ARG_OUTFILE(false, &mmse_file, "denoised samples (i.e. mmse estimate)"),
	};

	struct opt_s sigma_opts[] = {
		OPTL_FLOAT(0, "min", &sigma_min, "min", "minimum sigma for sampling"),
		OPTL_FLOAT(0, "max", &sigma_max, "max", "maximum sigma for sampling"),
		OPTL_SELECT(0, "exp", enum SIGMA_SCHEDULE, &schedule, SIGMA_SCHEDULE_EXP, "exponential decay of sigma"),
		OPTL_SELECT(0, "quad", enum SIGMA_SCHEDULE, &schedule, SIGMA_SCHEDULE_QUAD, "quadratic decay of sigma"),
	};

	struct opt_s gmm_opts[] = {
		OPTL_INFILE(0, "mean", &means_file, "file", "means of the Gaussians in the gmm"),
		OPTL_INFILE(0, "var", &vars_file, "file", "variance of the Gaussians in the gmm"),
		OPTL_INFILE(0, "w", &ws_file, "file", "weigthing of the Gaussians in the gmm"),
	};

	const struct opt_s opts[] = {
		OPTL_VECN(0, "dims", img_dims, "image dimensions"),
		OPT_SET('g', &bart_use_gpu, "use gpu"),
		OPT_UINT('s', &seed, "s", "seed"),
		OPTL_SET('r', "real-valued", &real_valued, "real-valued trained network (i.e. with z ~ CN(0, 2I))"),
		OPTL_SET('a', "ancestral", &ancestral, "ancestral sampling"),
		OPTL_SET('p', "predictor-corrector", &predictor_corrector, "predictor-corrector sampling"),
		OPTL_SUBOPT(0, "sigma", "", "select noise schedule for decreasing noise", ARRAY_SIZE(sigma_opts), sigma_opts),
		OPTL_SUBOPT(0, "gmm", "", "generate a Gaussian mixture model for sampling", ARRAY_SIZE(gmm_opts), gmm_opts),
		OPTL_STRING(0, "external-graph", &graph, "weights", ".pt or .tf file with weights"),
		OPTL_FLOAT(0, "gamma", &gamma_base, "gamma", "scaling of stepsize for Langevin iteration"),
		OPT_INT('N', &N, "N", "number of noise levels"),
		OPT_INT('K', &K, "K", "number of Langevin steps per level"),
		OPT_LONG('S', &batchsize, "S", "number of samples drawn"),
		OPTL_LONG(0, "save-mod", &save_mod, "S", "save samples every S steps"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init_gpu_support();

#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = bart_use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!bart_use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif
	sigma_schedule_fun_t get_sigma = NULL;	// false positive

	switch (schedule) {

	case SIGMA_SCHEDULE_EXP:
		get_sigma = sigma_schedule_exp;
		break;

	case SIGMA_SCHEDULE_QUAD:
		get_sigma = sigma_schedule_quad;
		break;
	}

	num_rand_init(seed);

	const struct nlop_s* nlop = NULL;
	img_dims[BATCH_DIM] = batchsize;

	float min_var = 0.0f;

	if (NULL != graph) {

		// generates nlop from tf or pt graph
		int DO[1] = { 3 };
		int DI[2] = { 3, 1 };
		long idims1[3] = { img_dims[0], img_dims[1], batchsize };
		long idims2[1] = { batchsize };

		nlop = nlop_external_graph_create(graph, 1, DO, (const long*[1]) { idims1 },  2, DI, (const long*[2]) {idims1, idims2}, bart_use_gpu, key);

		nlop = nlop_reshape_in_F(nlop, 0, DIMS, img_dims);
		nlop = nlop_reshape_out_F(nlop, 0, DIMS, img_dims);

		nlop = nlop_expectation_to_score(nlop);

		auto par = nlop_generic_domain(nlop, 1);

		if (1 < md_calc_size(par->N, par->dims))
			nlop = nlop_chain2_FF(nlop_from_linop_F(linop_repmat_create(par->N, par->dims, ~0ul)), 0, nlop, 1);

		nlop_unset_derivatives(nlop);

	} else if (NULL != means_file) {

		long means_dims[DIMS];
		long ws_dims[DIMS];
		long vars_dims[DIMS];

		complex float* means = load_cfl(means_file, DIMS, means_dims);

		img_dims[0] = means_dims[0];
		img_dims[1] = means_dims[1];
		img_dims[2] = means_dims[2];

		// check if ws are given, otherwise use uniform weights over all mean peaks
		complex float* ws = NULL;

		if (NULL == ws_file) {

			md_select_dims(DIMS, ~md_nontriv_dims(DIMS, img_dims), ws_dims, means_dims);

			ws = md_alloc_sameplace(DIMS, ws_dims, CFL_SIZE, means);

			long num_gaussians = md_calc_size(DIMS, ws_dims);
			md_zfill(DIMS, ws_dims, ws, 1. / num_gaussians);

			debug_printf(DP_WARN, "No weighting specified. Uniform weigths are set.\n");

		} else {

			ws = load_cfl(ws_file, DIMS, ws_dims);

			float wsum = md_zasum(DIMS, ws_dims, ws);
			md_zsmul(DIMS, ws_dims, ws, ws, 1. / wsum);
			
		}

		complex float* vars = NULL;

		if (NULL == vars_file) {

			md_copy_dims(DIMS, vars_dims, ws_dims);

			vars = md_alloc_sameplace(DIMS, vars_dims, CFL_SIZE, means);
			md_zfill(DIMS, vars_dims, vars, 0);
			debug_printf(DP_WARN, "No variance specified. Set to 0.\n");

		} else {

			vars = load_cfl(vars_file, DIMS, vars_dims);
		}

		assert(md_check_equal_dims(DIMS, means_dims, vars_dims, ~md_nontriv_dims(DIMS, img_dims)));
		assert(md_check_equal_dims(DIMS, means_dims, ws_dims, ~md_nontriv_dims(DIMS, img_dims)));

		// Find minimum element in vars
		long num_elements = md_calc_size(DIMS, vars_dims);
		min_var = crealf(vars[0]);
		for (long i = 1; i < num_elements; i++) {

			float v = crealf(vars[i]);

			if (v < min_var)
				min_var = v;
		}

		debug_printf(DP_DEBUG2, "Minimum variance in vars: %f\n", min_var);

		nlop = nlop_gmm_score_create(DIMS, img_dims, means_dims, means, vars_dims, vars, ws_dims, ws);

		ws_file ? unmap_cfl(DIMS, ws_dims, ws) : md_free(ws);
		vars_file ? unmap_cfl(DIMS, vars_dims, vars) : md_free(vars);
		means_file ? unmap_cfl(DIMS, means_dims, means) : md_free(means);

	} else {

		error("No network or gmm specified!\n");
	}

	nlop = nlop_reshape_in_F(nlop, 1, 1, (long[1]) { 1 }); // reshape noise scale from [1,1,1....1] to [1]

	if (real_valued) {

		nlop = nlop_prepend_FF(nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), sqrtf(0.5))), nlop, 1);
		sigma_max *= sqrtf(2.);
		sigma_min *= sqrtf(2.);
	}

	if (0 == save_mod)
		save_mod = N;

	assert(0 == N % save_mod);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, img_dims);
	out_dims[ITER_DIM] = N / save_mod;

	complex float* expectation = (mmse_file ? create_cfl : anon_cfl)(mmse_file, DIMS, out_dims);

	complex float* out = create_cfl(samples_file, DIMS, out_dims);
	long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

	complex float* samples = my_alloc(DIMS, img_dims, CFL_SIZE);

	md_gaussian_rand(DIMS, img_dims, samples);
	md_zsmul(DIMS, img_dims, samples, samples, 1 / sqrtf(2.)); // cplx var 1

	debug_printf(DP_DEBUG2, "sig=%.4f\n", get_sigma(1., sigma_min, sigma_max));
	md_zsmul(DIMS, img_dims, samples, samples, get_sigma(1., sigma_min, sigma_max));

	struct linop_s* linop = linop_null_create(DIMS, img_dims, DIMS, img_dims);

	complex float* AHy = my_alloc(DIMS, img_dims, CFL_SIZE);
	md_clear(DIMS, img_dims, AHy, CFL_SIZE);

	float maxeigen = 0.;


	for (int i = N - 1; i >= 0; i--) {

		float var_i = powf(get_sigma(((float)i) / N, sigma_min, sigma_max), 2);
		float var_ip = powf(get_sigma(((float)(i + 1)) / N, sigma_min, sigma_max), 2);
		float dvar = (var_ip - var_i);
		float tau_ip = var_i / var_ip * dvar;

		print_stats(DP_DEBUG2, (float)i / N, img_dims, samples, sqrtf(var_ip));

		if (ancestral || predictor_corrector) {

			complex float* tmp = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);

			complex float fixed_noise_scale = sqrtf(var_ip);
			const struct nlop_s* nlop_fixed = nlop_set_input_const(nlop, 1, 1, MD_DIMS(1), true, &fixed_noise_scale);
			nlop_apply(nlop_fixed, DIMS, img_dims, tmp, DIMS, img_dims, samples);
			nlop_free(nlop_fixed);

			md_zaxpy(DIMS, img_dims, samples, dvar / 2.f, tmp);

			md_gaussian_rand(DIMS, img_dims, tmp);
			md_zsmul(DIMS, img_dims, tmp, tmp, 1 / sqrtf(2.)); // cplx var 1

			md_zaxpy(DIMS, img_dims, samples, ancestral ? sqrtf(tau_ip) : sqrtf(dvar), tmp);

			md_free(tmp);

			if (ancestral)	// No Langevin steps if ancestral
				K = 0;
		}

		// Corrector
		complex float fixed_noise_scale = sqrtf(var_i);
		const struct nlop_s* nlop_fixed = nlop_set_input_const(nlop, 1, 1, MD_DIMS(1), true, &fixed_noise_scale);
		const struct operator_p_s* score_op_p = prox_nlgrad_create(nlop_fixed, 1, 1., -1, true); // convert grad to prox; mind the SIGN for the score

		score_op_p = prox_scale_arg_create_F(score_op_p, 0.5); // scale due to implementation of em (add 0.5 factor)

		float gamma = gamma_base / (1 / (var_i + min_var) + maxeigen);
		debug_printf(DP_DEBUG2, "gamma: %.5f\n", gamma);

		struct iter_eulermaruyama_conf em_conf = iter_eulermaruyama_defaults;
		em_conf.step = gamma;
		em_conf.maxiter = K;

		// run K Langevin steps
		iter2_eulermaruyama(CAST_UP(&em_conf), linop->normal, 1, &score_op_p, NULL, NULL, NULL, 2 * md_calc_size(DIMS,img_dims), (float*)samples, (float*)AHy, NULL);

		if (0 == i % save_mod) {

			pos[ITER_DIM] = i / save_mod;

			complex float* tmp_exp = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);

			// compute expectation value at current noise level
			nlop_apply(nlop_fixed, DIMS, img_dims, tmp_exp, DIMS, img_dims, samples);
			md_zsmul(DIMS, img_dims, tmp_exp, tmp_exp, var_i / 2);
			md_zaxpy(DIMS, img_dims, tmp_exp, 1, samples);

			md_copy_block(DIMS, pos, out_dims, expectation, img_dims, tmp_exp, CFL_SIZE);
			md_copy_block(DIMS, pos, out_dims, out, img_dims, samples, CFL_SIZE);

			md_free(tmp_exp);
		}

		operator_p_free(score_op_p);
		nlop_free(nlop_fixed);
	}

	print_stats(DP_DEBUG2, 0., img_dims, samples, get_sigma(0., sigma_min, sigma_max));

	nlop_free(nlop);
	linop_free(linop);

	md_free(AHy);

	md_free(samples);
	unmap_cfl(DIMS, out_dims, out);
	unmap_cfl(DIMS, out_dims, expectation);

	return 0;
}

