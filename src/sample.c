/* Copyright 2025-2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *          Tina Holliber
 *          Verena Fink
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/init.h"
#include "num/iovec.h"
#include "num/rand.h"
#include "num/ops_p.h"
#include "num/fft.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/mri2.h"

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
#include "networks/cunet.h"

#include "iter/iter2.h"
#include "iter/iter.h"
#include "iter/prox2.h"
#include "iter/misc.h"

#include "noncart/nufft.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] =
	"Prior sampling with given diffusion network (either PyTorch or TensorFlow) which is "
	"trained as denoiser (i.e. outputs the expectation) or Gaussian Mixture Model using "
	"unadjusted Langevin algorithm.\n";


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

	md_zstd2(DIMS, corn_dims, ~0UL, MD_SINGLETON_STRS(DIMS), std_device, MD_STRIDES(DIMS, img_dims, CFL_SIZE), samples);

	float std_corner;
	md_copy(1, MD_DIMS(1), &std_corner, std_device, FL_SIZE);

	complex float* tmp = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);

	fftuc(DIMS, img_dims, FFT_FLAGS, tmp, samples);
	md_zstd2(DIMS, corn_dims, ~0UL, MD_SINGLETON_STRS(DIMS), std_device, MD_STRIDES(DIMS, img_dims, CFL_SIZE), tmp);

	md_free(tmp);

	float kstd_corner;
	md_copy(1, MD_DIMS(1), &kstd_corner, std_device, FL_SIZE);

	md_free(std_device);

	debug_printf(dl, "t=%.2f; sig=%.4f; zstd/sig=%.2f; zstd(corner)/sig=%.2f; zstd(kspace corner)/sig=%.2f\n", t, sigma, std_samples / sigma, std_corner / sigma, kstd_corner / sigma);
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


static const struct linop_s* get_sense_linop(long img_dims[DIMS], long ksp_dims[DIMS], long col_dims[DIMS], complex float* sens, long traj_dims[DIMS], const complex float* traj, long pat_dims[DIMS], const complex float* pat)
{
	struct linop_s* lop_sense = NULL;

	long cim_dims[DIMS];
	md_max_dims(DIMS, ~0UL, cim_dims, img_dims, col_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, cim_dims);

	if (NULL != traj) {

		lop_sense = nufft_create2(DIMS, ksp_dims, cim_dims, traj_dims, traj, pat_dims, pat, MD_SINGLETON_DIMS(DIMS), NULL, nufft_conf_defaults);

	} else {

		lop_sense = linop_fftc_create(DIMS, col_dims, FFT_FLAGS);

		assert(md_check_equal_dims(DIMS, cim_dims, ksp_dims, ~0UL));
		assert(md_check_compat(DIMS, md_nontriv_dims(DIMS, ksp_dims), ksp_dims, pat_dims));

		lop_sense = linop_chain_FF(lop_sense, linop_cdiag_create(DIMS, ksp_dims, md_nontriv_dims(DIMS, pat_dims), pat));
	}

	return linop_chain_FF(linop_fmac_dims_create(DIMS, cim_dims, img_dims, col_dims, sens), lop_sense);
}


static void get_init(int N, long img_dims[N], complex float* samples, float sigma, const struct linop_s* A, const complex float* AHy, int iter)
{
	md_zgaussian_rand(N, img_dims, samples);
	md_zsmul(DIMS, img_dims, samples, samples, sigma);

	if ((NULL == A) || linop_is_null(A))
		return;

	const struct iovec_s* cod = linop_codomain(A);
	assert(N == cod->N);
	long ksp_dims[N];
	md_copy_dims(N, ksp_dims, cod->dims);

	complex float* tmp_ksp = md_alloc_sameplace(N, ksp_dims, CFL_SIZE, samples);

	md_zgaussian_rand(DIMS, ksp_dims, tmp_ksp);

	complex float* tmp_AHy = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);

	linop_adjoint(A, N, img_dims, tmp_AHy, N, ksp_dims, tmp_ksp);

	md_zadd(N, img_dims, tmp_AHy, tmp_AHy, AHy);

	md_zsmul(DIMS, img_dims, samples, samples, 1. / (sigma * sigma));
	md_zadd(N, img_dims, tmp_AHy, tmp_AHy, samples);
	md_zsmul(DIMS, img_dims, samples, samples, (sigma * sigma));

	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.maxiter = iter;
	conf.l2lambda = 1. / (sigma * sigma);

	iter2_conjgrad(CAST_UP(&conf), A->normal, 0, NULL, NULL, NULL, NULL, 2 * md_calc_size(DIMS, img_dims), (float*)(samples), (const float*)tmp_AHy, NULL);

	md_free(tmp_AHy);
	md_free(tmp_ksp);
}



int main_sample(int argc, char* argv[argc])
{
	const char* graph = NULL;
	const char* key = NULL;

	struct nn_cunet_conf_s cunet_conf = cunet_defaults;
	const char* cunet_weights = NULL;

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

	const char* kspace_file = NULL;
	const char* sens_file = NULL;
	const char* traj_file = NULL;
	const char* pattern_file = NULL;
	const char* mask_file = NULL;

	bool annealed = false;
	int precond_iter = -1;

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

	struct opt_s cunet_opts[] = {
		OPTL_INFILE('w', "weights", &cunet_weights, "weights", "weights for cunet"),
		OPTL_INT('l', "level", &cunet_conf.levels, "l", "Number of UNet levels"),
	};

	struct opt_s posterior_opts[] = {
		OPTL_INFILE('k', "kspace", &kspace_file, "file", "kspace file"),
		OPTL_INFILE('s', "sens", &sens_file, "file", "sensitivities"),
		OPTL_INFILE('t', "traj", &traj_file, "file", "k-space trajectory"),
		OPTL_INFILE('p', "pattern", &pattern_file, "file", "Pattern file"),
		OPTL_SET(0, "annealed", &annealed, "use annealed likelihood"),
		OPTL_INT(0, "precond", &precond_iter, "iter", "(number of preconditioning cg iterations)"),
		OPTL_INT(0, "precond_iter", &precond_iter, "iter", "number of preconditioning cg iterations"),
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
		OPTL_SUBOPT(0, "cunet", "", "sampling with conditional unet", ARRAY_SIZE(cunet_opts), cunet_opts),
		OPTL_STRING(0, "external-graph", &graph, "weights", ".pt or .tf file with weights"),
		OPTL_INFILE(0, "mask", &mask_file, "file", "FoV mask for output of network"),
		OPTL_FLOAT(0, "gamma", &gamma_base, "gamma", "scaling of stepsize for Langevin iteration"),
		OPT_INT('N', &N, "N", "number of noise levels"),
		OPT_INT('K', &K, "K", "number of Langevin steps per level"),
		OPT_LONG('S', &batchsize, "S", "number of samples drawn"),
		OPTL_LONG(0, "save-mod", &save_mod, "S", "save samples every S steps"),
		OPTL_SUBOPT(0, "posterior", "", "sample posterior", ARRAY_SIZE(posterior_opts), posterior_opts),
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

	if (annealed && (-1 == precond_iter))
		error("Preconditioning not supported for annealing.\n");

	const struct nlop_s* nlop = NULL;
	const struct linop_s* linop = NULL;

	bool posterior = (NULL != kspace_file);

	long ksp_dims[DIMS];
	complex float* ksp = NULL;

	if (posterior) {

		ksp = load_cfl(kspace_file, DIMS, ksp_dims);

		long map_dims[DIMS];
		complex float* sens = load_cfl(sens_file, DIMS, map_dims);
		md_select_dims(DIMS, ~COIL_FLAG, img_dims, map_dims);

		long trj_dims[DIMS];
		complex float* traj = NULL;

		if (traj_file)
			traj = load_cfl(traj_file, DIMS, trj_dims);

		long pat_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		complex float* pat;

		if (NULL == pattern_file) {

			pat = anon_cfl(NULL, DIMS, pat_dims);

			estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pat, ksp);

		} else {

			pat = load_cfl(pattern_file, DIMS, pat_dims);
		}

		linop = get_sense_linop(img_dims, ksp_dims, map_dims, sens, trj_dims, traj, pat_dims, pat);

		unmap_cfl(DIMS, map_dims, sens);
		unmap_cfl(DIMS, pat_dims, pat);

		if (NULL != traj_file)
			unmap_cfl(DIMS, trj_dims, traj);

		if (-1 == precond_iter)
			precond_iter = 10;
	}

	img_dims[BATCH_DIM] = batchsize;

	float min_var = 0.;

	if (NULL != graph || NULL != cunet_weights) {

		if (NULL != cunet_weights) {

			const long dims[5] = { 1, img_dims[0], img_dims[1], img_dims[2], batchsize };

			nn_t cunet = cunet_create(&cunet_conf, 5, dims);

			cunet = nn_denoise_precond_edm(cunet, -1., -1., 0.5, false);

			nn_weights_t weights = load_nn_weights(cunet_weights);

			nlop = nn_get_nlop_wo_weights_F(cunet, weights, true);

			nn_weights_free(weights);

		} else {

			// generates nlop from tf or pt graph
			int DO[1] = { 3 };
			int DI[2] = { 3, 1 };
			long idims1[3] = { img_dims[0], img_dims[1], batchsize };
			long idims2[1] = { batchsize };

			nlop = nlop_external_graph_create(graph, 1, DO, (const long*[1]) { idims1 }, 
					2, DI, (const long*[2]) {idims1, idims2}, bart_use_gpu, key);
		}

		nlop = nlop_reshape_in_F(nlop, 0, DIMS, img_dims);
		nlop = nlop_reshape_out_F(nlop, 0, DIMS, img_dims);

		auto par = nlop_generic_domain(nlop, 1);

		if (1 < md_calc_size(par->N, par->dims))
			nlop = nlop_chain2_FF(nlop_from_linop_F(linop_repmat_create(par->N, par->dims, ~0UL)), 0, nlop, 1);

		if (real_valued)
			nlop = nlop_prepend_FF(nlop_from_linop_F(linop_scale_create(1, MD_DIMS(1), sqrtf(0.5))), nlop, 1);

		if (NULL != mask_file) {

			long msk_dims[DIMS];
			complex float* msk = load_cfl(mask_file, DIMS, msk_dims);

			assert(md_check_equal_dims(DIMS, msk_dims, img_dims, md_nontriv_dims(DIMS, msk_dims)));

			const struct linop_s* lop_msk = linop_cdiag_create(DIMS, img_dims, md_nontriv_dims(DIMS, msk_dims), msk);

			nlop = nlop_append_FF(nlop, 0, nlop_from_linop_F(lop_msk));

			unmap_cfl(DIMS, msk_dims, msk);
		}

		nlop = nlop_expectation_to_score(nlop);

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


	if (0 == save_mod)
		save_mod = N;

	assert(0 == N % save_mod);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, img_dims);
	out_dims[ITER_DIM] = N / save_mod;

	complex float* expectation = (mmse_file ? create_cfl : anon_cfl)(mmse_file, DIMS, out_dims);

	complex float* out = create_cfl(samples_file, DIMS, out_dims);
	long pos[DIMS] = { };

	complex float* samples = my_alloc(DIMS, img_dims, CFL_SIZE);

	complex float* AHy = my_alloc(DIMS, img_dims, CFL_SIZE);
	md_clear(DIMS, img_dims, AHy, CFL_SIZE);

	float maxeigen = 0.;

	if (posterior) {

		if (0 == precond_iter)
			maxeigen = estimate_maxeigenval_sameplace(linop->normal, 30, samples);

		long img_single_dims[DIMS];
		md_select_dims(DIMS, ~BATCH_FLAG, img_single_dims, img_dims);

		complex float* tmp_AHy = md_alloc_sameplace(DIMS, img_single_dims, CFL_SIZE, ksp);

		linop_adjoint(linop, DIMS, img_single_dims, tmp_AHy, DIMS, ksp_dims, ksp);

		md_copy2(DIMS, img_dims, MD_STRIDES(DIMS, img_dims, CFL_SIZE), AHy, MD_STRIDES(DIMS, img_single_dims, CFL_SIZE), tmp_AHy, CFL_SIZE);
		md_free(tmp_AHy);

		long loop_dims[DIMS];
		md_select_dims(DIMS, BATCH_FLAG, loop_dims, img_dims);

		linop = linop_loop_F(DIMS, loop_dims, (struct linop_s*)linop);

	} else {

		linop = linop_null_create(DIMS, img_dims, DIMS, img_dims);
	}

	get_init(DIMS, img_dims, samples, get_sigma(1., sigma_min, sigma_max),
		 (0 < precond_iter) ? linop : NULL, AHy, precond_iter);

	for (int i = N - 1; i >= 0; i--) {

		float var_i = powf(get_sigma(((float)i) / N, sigma_min, sigma_max), 2.);
		float var_ip = powf(get_sigma(((float)(i + 1)) / N, sigma_min, sigma_max), 2.);
		float dvar = (var_ip - var_i);
		float tau_ip = var_i / var_ip * dvar;

		print_stats(DP_DEBUG2, (float)i / N, img_dims, samples, sqrtf(var_ip));

		float maxeigen_iter = maxeigen;
		const struct linop_s* linop_iter = linop_clone(linop);
		complex float* AHy_iter = AHy;

		if (annealed) {

			float scale = get_sigma(((float)i) / N, 1., 1./ sigma_max / sqrtf(maxeigen));

			AHy_iter = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);

			md_zsmul(DIMS, img_dims, AHy_iter, AHy, powf(scale, 2.));

			maxeigen_iter *= powf(scale, 2.);

			linop_iter = linop_chain_FF(linop_scale_create(DIMS, img_dims, scale), linop_iter);
		}


		if (ancestral || predictor_corrector) {

			complex float* tmp = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);

			complex float fixed_noise_scale = sqrtf(var_ip);
			const struct nlop_s* nlop_fixed = nlop_set_input_const(nlop, 1, 1, MD_DIMS(1), true, &fixed_noise_scale);

			nlop_apply(nlop_fixed, DIMS, img_dims, tmp, DIMS, img_dims, samples);

			nlop_free(nlop_fixed);

			if (posterior) {

				complex float* tmp1 = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);

				linop_normal(linop_iter, DIMS, img_dims, tmp1, samples);

				md_zsub(DIMS, img_dims, tmp1, AHy_iter, tmp1);
				md_zadd(DIMS, img_dims, tmp, tmp, tmp1);

				md_free(tmp1);
			}

			md_zaxpy(DIMS, img_dims, samples, dvar, tmp);

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

		float gamma = gamma_base / (1 / (var_i + min_var) + maxeigen_iter);

		// run K Langevin steps
		struct iter_eulermaruyama_conf em_conf = iter_eulermaruyama_defaults;
		em_conf.step = gamma;
		em_conf.maxiter = K;

		if (0 < precond_iter) {

			em_conf.step = gamma_base;
			em_conf.maxiter = K;
			em_conf.precond_linop = linop_iter;
			em_conf.precond_max_iter = precond_iter;
			em_conf.precond_diag = 1. / var_i;
			em_conf.batchsize = batchsize;
		}

		debug_printf(DP_DEBUG2, "gamma: %.2e\n", em_conf.step);

		iter2_eulermaruyama(CAST_UP(&em_conf), linop_iter->normal, 1, &score_op_p,
				NULL, NULL, NULL, 2 * md_calc_size(DIMS,img_dims),
				(float*)samples, (float*)AHy_iter, NULL);

		if (0 == i % save_mod) {

			pos[ITER_DIM] = i / save_mod;

			complex float* tmp_exp = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);

			// compute expectation value at current noise level
			nlop_apply(nlop_fixed, DIMS, img_dims, tmp_exp, DIMS, img_dims, samples);

			md_zsmul(DIMS, img_dims, tmp_exp, tmp_exp, var_i);
			md_zaxpy(DIMS, img_dims, tmp_exp, 1., samples);

			md_copy_block(DIMS, pos, out_dims, expectation, img_dims, tmp_exp, CFL_SIZE);
			md_copy_block(DIMS, pos, out_dims, out, img_dims, samples, CFL_SIZE);

			md_free(tmp_exp);
		}

		operator_p_free(score_op_p);
		nlop_free(nlop_fixed);
		linop_free(linop_iter);

		if (AHy_iter != AHy)
			md_free(AHy_iter);
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

