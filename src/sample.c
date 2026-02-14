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
#include "misc/stream.h"

#include "noncart/nufft.h"

#include "linops/linop.h"
#include "linops/sum.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/gmm.h"

#include "iter/iter2.h"
#include "iter/iter.h"
#include "iter/prox.h"
#include "iter/prox2.h"
#include "iter/misc.h"

#include "networks/cunet.h"

#include "grecon/model.h"
#include "grecon/priors.h"

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





static void get_init(int N, long img_dims[N], complex float* samples, float sigma, const struct linop_s* A, const complex float* AHy, struct iter_eulermaruyama_conf em_conf)
{
	if ((NULL == A) || linop_is_null(A)) {

		md_zgaussian_rand(N, img_dims, samples);
		md_zsmul(DIMS, img_dims, samples, samples, sigma);

	} else {

		// one pULA step, initialized with zero, without a prior score, and stepsize 2, yields:
		// x = 2 (A^HA + I / (2 sigma^2))^-1 [ A^H y + n1 + 1 / sigma n2] with n1, n2 ~ CN(0, I)
		// This is twixe a sample from the posterior distribution
		// p(x|y) ~ exp(-||y - A x||^2) exp(-1/sigma^2||x||^2)

		em_conf.maxiter = 1;
		em_conf.step = 2;
		em_conf.precond_linop = A;
		em_conf.precond_diag = 1. / (sigma * sigma);

		// we use prox zero here, as, applied to the initial zero it returns zero
		const struct operator_p_s* t_prox = prox_zero_create(N, img_dims);

		md_clear(N, img_dims, samples, CFL_SIZE);

		iter2_eulermaruyama(CAST_UP(&em_conf), A->normal, 1, &t_prox,
				NULL, NULL, NULL, 2 * md_calc_size(DIMS, img_dims),
				(float*)samples, (float*)AHy, NULL);

		md_zsmul(N, img_dims, samples, samples, 0.5);

		operator_p_free(t_prox);
	}
}



int main_sample(int argc, char* argv[argc])
{
	const char* graph = NULL;

	struct nn_cunet_conf_s cunet_conf = cunet_defaults;
	const char* cunet_weights = NULL;

	const char* means_file = NULL;
	const char* vars_file = NULL;
	const char* ws_file = NULL;

	const char* samples_file = NULL;
	const char* mmse_file = NULL;
	unsigned int seed = 123;

	float sigma_min = 0.01;
	float sigma_max = 10.;


	enum SIGMA_SCHEDULE { SIGMA_SCHEDULE_EXP, SIGMA_SCHEDULE_QUAD };
	enum SIGMA_SCHEDULE schedule = SIGMA_SCHEDULE_EXP;

	struct iter_eulermaruyama_conf em_conf = iter_eulermaruyama_defaults;
	em_conf.precond_max_iter = -1;
	em_conf.maxiter = 1;
	em_conf.batchsize = 1;

	int N = 100;
	bool ancestral = false;
	bool predictor_corrector = false;
	bool real_valued = false;
	bool dps = false;

	long save_mod = 0;

	float gamma_base = 0.5;

	const char* kspace_file = NULL;
	const char* sens_file = NULL;
	const char* traj_file = NULL;
	const char* pattern_file = NULL;
	const char* mask_file = NULL;

	bool annealed = false;

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
		OPTL_SET(0, "dps", &dps, "use DPS sampling (predictor only)"),
		OPTL_INT(0, "precond", &em_conf.precond_max_iter, "iter", "(number of preconditioning cg iterations)"),
		OPTL_INT(0, "precond_iter", &em_conf.precond_max_iter, "iter", "number of preconditioning cg iterations"),
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
		OPT_INT('K', &em_conf.maxiter, "K", "number of Langevin steps per level"),
		OPT_LONG('S', &em_conf.batchsize, "S", "number of samples drawn"),
		OPTL_LONG(0, "save-mod", &save_mod, "S", "save samples every S steps"),
		OPTL_SUBOPT(0, "posterior", "", "sample posterior", ARRAY_SIZE(posterior_opts), posterior_opts),
		OPTL_SUBOPT(0, "nufft-conf", "...", "configure nufft", N_nufft_conf_opts, nufft_conf_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init_gpu_support();

#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = bart_use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!bart_use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	NESTED(float, get_sigma0, (int i, float sigma_min, float sigma_max))
	{
		float t = (float)i / N;

		switch (schedule) {

		case SIGMA_SCHEDULE_EXP:
			return sigma_min * expf(logf(sigma_max / sigma_min) * t);
			break;

		case SIGMA_SCHEDULE_QUAD:
			return sigma_min + sigma_max * t * t;
			break;

		default:
			unreachable();
		}
	};

	NESTED(float, get_sigma, (int i))
	{
		return get_sigma0(i, sigma_min, sigma_max);
	};

	num_rand_init(seed);

	if ((annealed || dps) && (-1 == em_conf.precond_max_iter))
		em_conf.precond_max_iter = 0;

	if ((annealed || dps) && !(0 == em_conf.precond_max_iter))
		error("Preconditioning not supported for annealing or DPS.\n");

	if (ancestral)	// No Langevin steps if ancestral
		em_conf.maxiter = 0;


	const struct linop_s* linop = NULL;

	bool posterior = (NULL != kspace_file);

	long ksp_dims[DIMS];
	complex float* ksp = NULL;

	long pat_dims[DIMS];
	complex float* pat = NULL;

	long map_dims[DIMS];
	complex float* sens = NULL;

	complex float yHy = 0.;

	if (posterior) {

		ksp = load_cfl(kspace_file, DIMS, ksp_dims);
		yHy = crealf(md_zscalar(DIMS, ksp_dims, ksp, ksp));

		sens = load_cfl(sens_file, DIMS, map_dims);
		md_select_dims(DIMS, ~COIL_FLAG, img_dims, map_dims);

		long trj_dims[DIMS];
		complex float* traj = NULL;

		if (traj_file)
			traj = load_cfl(traj_file, DIMS, trj_dims);

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);

		if (NULL == pattern_file) {

			pat = anon_cfl(NULL, DIMS, pat_dims);

			estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pat, ksp);

		} else {

			pat = load_cfl(pattern_file, DIMS, pat_dims);
		}

		if (NULL == traj)
			ifftmod(DIMS, ksp_dims, FFT_FLAGS, ksp, ksp);

		struct pics_config conf = { };
		conf.gpu = bart_use_gpu;
		conf.nuconf = &nufft_conf_options;

		linop = pics_model(&conf, img_dims, ksp_dims, trj_dims, traj, NULL, NULL,
				   map_dims, sens, pat_dims, pat, NULL, NULL, NULL);

		unmap_cfl(DIMS, trj_dims, traj);

		if (-1 == em_conf.precond_max_iter)
			em_conf.precond_max_iter = 10;
	}

	img_dims[BATCH_DIM] = em_conf.batchsize;

	float min_var = 0.;

	long msk_dims[DIMS];
	complex float* msk = NULL;

	if (NULL != mask_file)
		msk = load_cfl(mask_file, DIMS, msk_dims);

	const struct nlop_s* nlop = NULL;

	if (NULL != graph) {

		nlop = prior_graph(graph, real_valued, bart_use_gpu, msk_dims, msk, img_dims);

	} else if (NULL != cunet_weights) {

		nlop = prior_cunet(cunet_weights, &cunet_conf, real_valued,
					msk_dims, msk, img_dims);

	} else if (NULL != means_file) {

		long means_dims[DIMS];
		complex float* means = load_cfl(means_file, DIMS, means_dims);

		long weights_dims[DIMS];
		const complex float* weights = NULL;

		if (NULL != ws_file)
			weights = load_cfl(ws_file, DIMS, weights_dims);

		long vars_dims[DIMS];
		const complex float* vars = NULL;

		if (NULL != vars_file)
			vars = load_cfl(vars_file, DIMS, vars_dims);

		nlop = prior_gmm(means_dims, means, weights_dims, weights, vars_dims, vars, img_dims, &min_var);

		unmap_cfl(DIMS, weights_dims, weights);
		unmap_cfl(DIMS, vars_dims, vars);
		unmap_cfl(DIMS, means_dims, means);

	} else {

		error("No network or gmm specified!\n");
	}

	if (!dps)
		nlop_unset_derivatives(nlop);

	unmap_cfl(DIMS, msk_dims, msk);

	nlop = nlop_reshape_in_F(nlop, 1, 1, (long[1]) { 1 }); // reshape noise scale from [1,1,1....1] to [1]

	if (0 == save_mod)
		save_mod = N;

	assert(0 == N % save_mod);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, img_dims);
	out_dims[ITER_DIM] = N / save_mod;

	complex float* expectation = (mmse_file ? create_cfl : anon_cfl)(mmse_file, DIMS, out_dims);

	complex float* out = create_async_cfl(samples_file, MD_BIT(ITER_DIM), DIMS, out_dims);
	long pos[DIMS] = { };

	stream_t strm_o = stream_lookup(out);

	complex float* samples = my_alloc(DIMS, img_dims, CFL_SIZE);

	complex float* AHy = my_alloc(DIMS, img_dims, CFL_SIZE);
	md_clear(DIMS, img_dims, AHy, CFL_SIZE);

	float maxeigen = 0.;

	if (posterior) {

		if (0 == em_conf.precond_max_iter)
			maxeigen = estimate_maxeigenval_sameplace(linop->normal, 30, samples);

		long img_single_dims[DIMS];
		md_select_dims(DIMS, ~BATCH_FLAG, img_single_dims, img_dims);

		complex float* tmp_AHy = md_alloc_sameplace(DIMS, img_single_dims, CFL_SIZE, ksp);

		linop_adjoint(linop, DIMS, img_single_dims, tmp_AHy, DIMS, ksp_dims, ksp);

		md_copy2(DIMS, img_dims, MD_STRIDES(DIMS, img_dims, CFL_SIZE), AHy,
					 MD_STRIDES(DIMS, img_single_dims, CFL_SIZE), tmp_AHy, CFL_SIZE);

		md_free(tmp_AHy);

		long loop_dims[DIMS];
		md_select_dims(DIMS, BATCH_FLAG, loop_dims, img_dims);

		linop = linop_loop_F(DIMS, loop_dims, (struct linop_s*)linop);

	} else {

		linop = linop_null_create(DIMS, img_dims, DIMS, img_dims);
	}

	if (dps) {

		if ((0 != em_conf.precond_max_iter) || (!predictor_corrector) || ancestral || !posterior)
			error("DPS is only supported with predictor sampling (K = 0)!\n");
	}

	get_init(DIMS, img_dims, samples, sigma_max,
		 (0 < em_conf.precond_max_iter) ? linop : NULL, AHy, em_conf);

	for (int i = N - 1; i >= 0; i--) {

		float var_i = powf(get_sigma(i), 2.);
		float var_ip = powf(get_sigma(i + 1), 2.);
		float dvar = (var_ip - var_i);
		float tau_ip = (var_i / var_ip) * dvar;

		print_stats(DP_DEBUG2, (float)i / N, img_dims, samples, sqrtf(var_ip));

		float maxeigen_iter = maxeigen;
		const struct linop_s* linop_iter = linop_clone(linop);
		complex float* AHy_iter = AHy;

		if (annealed) {

			float scale = get_sigma0(i, 1., 1. / sigma_max / sqrtf(maxeigen));

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

			if (posterior) {

				complex float* tmp1 = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);
				complex float* tmp2 = NULL;

				if (dps) {

					tmp2 = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, samples);
					md_zsmul(DIMS, img_dims, tmp2, tmp, var_ip);
					md_zadd(DIMS, img_dims, tmp2, tmp2, samples);
				}

				linop_normal(linop_iter, DIMS, img_dims, tmp1, tmp2 ?: samples);

				md_zsub(DIMS, img_dims, tmp1, AHy_iter, tmp1);

				if (dps) {

					long bdims[DIMS];
					md_select_dims(DIMS, BATCH_FLAG, bdims, img_dims);

					complex float* nrm = md_alloc_sameplace(DIMS, bdims, CFL_SIZE, tmp1);
					md_ztenmulc(DIMS, bdims, nrm, img_dims, tmp2, img_dims, AHy_iter); 	//   (AHy)^H * x
					md_zfmacc2(DIMS, img_dims, MD_STRIDES(DIMS, bdims, CFL_SIZE), nrm,
						   MD_STRIDES(DIMS, img_dims, CFL_SIZE), tmp1,
						   MD_STRIDES(DIMS, img_dims, CFL_SIZE), tmp2);	// -x^H(AHAx - AHy) + (AHy)^H * x
					md_zsmul(DIMS, bdims, nrm, nrm, -1);			// (Ax)^H Ax - (Ax)^H y - (y)^H * Ax
					md_zsadd(DIMS, bdims, nrm, nrm, yHy);			// (Ax)^H Ax - (Ax)^H y - (y)^H * Ax + y^H y = ||y - Ax||_2^2
					md_zreal(DIMS, bdims, nrm, nrm);
					md_zspow(DIMS, bdims, nrm, nrm, -0.5);			// 1. / ||y - Ax||_2

					complex float* tmp = md_alloc(DIMS, bdims, CFL_SIZE);
					md_copy(DIMS, bdims, tmp, nrm, CFL_SIZE);

					float norm = 0.;
					for (int j = 0; j < bdims[BATCH_DIM]; j++)
						norm += bdims[BATCH_DIM] / crealf(tmp[j]);

					md_free(tmp);

					debug_printf(DP_DEBUG2, "        DPS: norm: %.2e, relative likelihood weighting: %.2e\n", norm, 0. >= gamma_base ? 1 : gamma_base / norm / dvar);

					md_zsmul(DIMS, bdims, nrm, nrm, 1. / dvar);
					md_zsmul(DIMS, bdims, nrm, nrm, gamma_base);

					if (0. >= gamma_base)
						md_zfill(DIMS, bdims, nrm, 1.);

					nlop_adjoint(nlop_fixed, DIMS, img_dims, tmp2, DIMS, img_dims, tmp1);
					md_zsmul(DIMS, img_dims, tmp2, tmp2, var_ip);
					md_zadd(DIMS, img_dims, tmp2, tmp2, tmp1);

					md_zmul2(DIMS, img_dims,
						 MD_STRIDES(DIMS, img_dims, CFL_SIZE), tmp2,
						 MD_STRIDES(DIMS, img_dims, CFL_SIZE), tmp2,
						 MD_STRIDES(DIMS, bdims, CFL_SIZE), nrm);		// 1 / ||y - Ax||_2 * score(x)

					md_free(nrm);
				}

				md_zadd(DIMS, img_dims, tmp, tmp, tmp2 ?: tmp1);

				md_free(tmp1);
				md_free(tmp2);
			}

			nlop_free(nlop_fixed);

			md_zaxpy(DIMS, img_dims, samples, dvar, tmp);

			md_zgaussian_rand(DIMS, img_dims, tmp);

			md_zaxpy(DIMS, img_dims, samples, ancestral ? sqrtf(tau_ip) : sqrtf(dvar), tmp);

			md_free(tmp);
		}

		// Corrector
		complex float fixed_noise_scale = sqrtf(var_i);
		const struct nlop_s* nlop_fixed = nlop_set_input_const(nlop, 1, 1, MD_DIMS(1), true, &fixed_noise_scale);

		// convert grad to prox; mind the SIGN for the score
		const struct operator_p_s* score_op_p[1] = { prox_nlgrad_create(nlop_fixed, 1, 1., -1, true) };

		if (0 >= em_conf.precond_max_iter) {

			em_conf.step = gamma_base / (1. / (var_i + min_var) + maxeigen_iter);

		} else {

			em_conf.step = gamma_base;
			em_conf.precond_linop = linop_iter;
			em_conf.precond_diag = 1. / var_i;
		}

		if (0 < em_conf.maxiter)
			debug_printf(DP_DEBUG2, "gamma: %.2e\n", em_conf.step);

		iter2_eulermaruyama(CAST_UP(&em_conf), linop_iter->normal, 1, score_op_p,
				NULL, NULL, NULL, 2 * md_calc_size(DIMS, img_dims),
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

			if (strm_o)
				stream_sync_slice(strm_o, DIMS, out_dims, MD_BIT(ITER_DIM), pos);
		}

		operator_p_free(score_op_p[0]);
		nlop_free(nlop_fixed);
		linop_free(linop_iter);

		if (AHy_iter != AHy)
			md_free(AHy_iter);
	}

	print_stats(DP_DEBUG2, 0., img_dims, samples, get_sigma(0));

	nlop_free(nlop);
	linop_free(linop);

	md_free(AHy);
	md_free(samples);

	unmap_cfl(DIMS, pat_dims, pat);
	unmap_cfl(DIMS, ksp_dims, ksp);
	unmap_cfl(DIMS, map_dims, sens);

	unmap_cfl(DIMS, out_dims, out);
	unmap_cfl(DIMS, out_dims, expectation);

	return 0;
}

