/* Copyright 2022. Institute of Medical Engineering. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <assert.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "iter/iter3.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"

#include "noir/model.h"
#include "noir/recon.h"

#include "moba/model_T1.h"
#include "moba/model_T2.h"
#include "moba/model_moba.h"
#include "moba/blochfun.h"
#include "moba/T1phyfun.h"
#include "moba/iter_l1.h"
#include "moba/moba.h"
#include "moba/exp.h"
#include "moba/recon_meco.h"

#include "recon.h"


static void post_process(enum mdb_t mode, const struct linop_s* op, struct moba_conf_s* data, const long dims[DIMS], complex float* img)
{
	long imgs_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, imgs_dims, dims);

	long pos[DIMS] = { 0L };

	// Project B1 map back into image space

        long map_dims[DIMS];
        md_select_dims(DIMS, FFT_FLAGS|TIME_FLAG|TIME2_FLAG, map_dims, dims);

	complex float* tmp = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, img);

	if (MDB_BLOCH == mode) {

                assert(NULL != data);

                pos[COEFF_DIM] = 3;

                md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);
                bloch_forw_alpha(op, tmp, tmp);
                md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);
        }

        // Reparameterized Look-Locker Model
        // Estimate effective flip angle from R1'
	// FIXME: Move to separate function which can be tested with a unit test

	if (MDB_T1_PHY == mode) {

		md_set_dims(DIMS, pos, 0);

		pos[COEFF_DIM] = 2;

                long map_size = md_calc_size(DIMS, map_dims);

		md_copy_block(DIMS, pos, map_dims, tmp, imgs_dims, img, CFL_SIZE);

		T1_forw_alpha(op, tmp, tmp);

		md_zreal(DIMS, map_dims, tmp, tmp);

		md_zsmul(DIMS, map_dims, tmp, tmp, -data->sim.seq.tr * 0.2);    // 0.2 -> Same scaling set in T1phyfun.c

		md_smin(1, MD_DIMS(2 * map_size), (float*)tmp, (float*)tmp, 0.);

		md_zexp(DIMS, map_dims, tmp, tmp);

		md_zacos(DIMS, map_dims, tmp, tmp);

	        md_zsmul(DIMS, map_dims, tmp, tmp, 180. / M_PI);        // output the effective flip angle map (in degree!)

		md_copy_block(DIMS, pos, imgs_dims, img, map_dims, tmp, CFL_SIZE);
	}

	md_free(tmp);
}


static void set_bloch_conf(enum mdb_t mode, struct mdb_irgnm_l1_conf* conf2, struct moba_conf_s* data)
{

	// T2 estimation turned off for IR FLASH Simulation

        if (MDB_BLOCH == mode) {

                assert(NULL != data);

                if (SEQ_IRFLASH == data->sim.seq.seq_type) {

                        conf2->constrained_maps = 1;	// only R1 map: bitmask (1 0 0 0) = 1
                        conf2->not_wav_maps = 2;	// no wavelet for T2 and B1 map
                }
                else if (SEQ_IRBSSFP == data->sim.seq.seq_type) {

                        conf2->constrained_maps = 5;	// only T1 and T2: bitmask(1 0 1 0) = 5
                        conf2->not_wav_maps = 1;	// no wavelet for B1 map
                }
        }

        // No Wavelet penalty on flip angle map

        if (MDB_T1_PHY == mode) {

                conf2->constrained_maps = 2;     // only R1 map: bitmask (0 1 0) = 2
                conf2->not_wav_maps = 1;
        }

	conf2->tvscales_N = data->other.tvscales_N;
	conf2->tvscales = data->other.tvscales;
}



static struct mobamod exp_create(const long dims[DIMS], const complex float* mask, const complex float* TE, const complex float* psf, const struct noir_model_conf_s* conf)
{
	long data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_s nlinv = noir_create3(data_dims, mask, psf, conf);
	struct mobamod ret;

	assert(2 == dims[COEFF_DIM]);

	long edims[DIMS];
	md_select_dims(DIMS, TE_FLAG, edims, dims);

	complex float* TE2 = md_alloc(DIMS, edims, CFL_SIZE);

	md_zsmul(DIMS, edims, TE2, TE, -10.);


	// chain T2 model
	struct nlop_s* a = nlop_exp_create(DIMS, data_dims, TE2);


	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2_FF(a, 0, b, 0);


	nlinv.nlop = nlop_permute_inputs_F(c, 3, (const int[3]){ 1, 2, 0 });

	ret.nlop = nlop_flatten(nlop_attach(nlinv.nlop, TE2, md_free));
	ret.linop = nlinv.linop;

	nlop_free(nlinv.nlop);

	return ret;
}



static void recon(const struct moba_conf* conf, struct moba_conf_s* data,
                const long dims[DIMS],
		const long imgs_dims[DIMS], complex float* img,
		const long coil_dims[DIMS], complex float* sens,
		const complex float* pattern,
		const complex float* mask,
		const complex float* TI,
		const complex float* b1,
		const long data_dims[DIMS], const complex float* kspace_data, bool usegpu)
{
	unsigned int fft_flags = FFT_FLAGS;

	if (conf->sms)
		fft_flags |= SLICE_FLAG;

	long img1_dims[DIMS];
	md_select_dims(DIMS, fft_flags, img1_dims, dims);


	struct noir_model_conf_s mconf = noir_model_conf_defaults;
	mconf.rvc = false;
	mconf.noncart = conf->noncartesian;
	mconf.fft_flags = fft_flags;
	mconf.a = conf->sobolev_a;
	mconf.b = conf->sobolev_b;
	mconf.cnstcoil_flags = TE_FLAG;

	struct mobamod nl = { 0 };

	switch (conf->mode) {

	case MDB_T1:

		nl = T1_create(dims, mask, TI, pattern, conf->scaling_M0, conf->scaling_R1s, &mconf, data->other.fov_reduction_factor);
		break;

	case MDB_T2:

#if 0
		// slower
		nl = exp_create(dims, mask, TI, pattern, &mconf);
#else
		UNUSED(exp_create);
		nl = T2_create(dims, mask, TI, pattern, &mconf, usegpu);
#endif
		break;

	case MDB_MGRE:

		assert(0);

        case MDB_T1_PHY:
        case MDB_BLOCH:

                nl = moba_create(dims, mask, TI, b1, pattern, &mconf, data, usegpu);
                break;
	}



	long skip = md_calc_size(DIMS, imgs_dims);
	long size = skip + md_calc_size(DIMS, coil_dims);
	long data_size = md_calc_size(DIMS, data_dims);

	long d1[1] = { size };
	// variable which is optimized by the IRGNM
	complex float* x = md_alloc_sameplace(1, d1, CFL_SIZE, kspace_data);

	md_copy(DIMS, imgs_dims, x, img, CFL_SIZE);
	md_copy(DIMS, coil_dims, x + skip, sens, CFL_SIZE);




	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;

	irgnm_conf.iter = conf->iter;
	irgnm_conf.alpha = conf->alpha;
	irgnm_conf.redu = conf->redu;

	if (conf->alpha_min_exp_decay)
		irgnm_conf.alpha_min = conf->alpha_min;
	else
		irgnm_conf.alpha_min0 = conf->alpha_min;

	irgnm_conf.cgtol = conf->tolerance;

	if (MDB_T1 == conf->mode) {

		if ((2 == conf->opt_reg) || (!conf->auto_norm))
			irgnm_conf.cgtol = 1e-3;
	}

	irgnm_conf.cgiter = conf->inner_iter;
	irgnm_conf.nlinv_legacy = true;

	struct mdb_irgnm_l1_conf conf2 = {

		.c2 = &irgnm_conf,
		.opt_reg = conf->opt_reg,
		.step = conf->step,
		.lower_bound = conf->lower_bound,
		.l2flags = (1 == conf->opt_reg) ? (0UL) : ~(0UL),
		.constrained_maps = 1UL << (dims[COEFF_DIM] - 1), // Always constrain last parameter map as default
		.auto_norm = conf->auto_norm,
                .not_wav_maps = 0,
		.algo = conf->algo,
		.rho = conf->rho,
		.ropts = conf->ropts
	};

        set_bloch_conf(conf->mode, &conf2, data);

	long irgnm_conf_dims[DIMS];
	md_select_dims(DIMS, fft_flags|MAPS_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, irgnm_conf_dims, imgs_dims);

	irgnm_conf_dims[COIL_DIM] = coil_dims[COIL_DIM];

	debug_printf(DP_INFO, "imgs_dims:\n\t");
	debug_print_dims(DP_INFO, DIMS, irgnm_conf_dims);


	mdb_irgnm_l1(&conf2,
			irgnm_conf_dims,
			nl.nlop,
			size * 2, (float*)x,
			data_size * 2, (const float*)kspace_data);

	md_copy(DIMS, imgs_dims, img, x, CFL_SIZE);

	if (NULL != sens) {

		noir_forw_coils(nl.linop, x + skip, x + skip);
		md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);
		fftmod(DIMS, coil_dims, fft_flags, sens, sens);
	}

	post_process(conf->mode, nl.linop_alpha, data, dims, img);

	// Clean up

        if ((MDB_T1_PHY == conf->mode) || (MDB_BLOCH == conf->mode))
                linop_free(nl.linop_alpha);

	nlop_free(nl.nlop);

	md_free(x);
}


void moba_recon(const struct moba_conf* conf, struct moba_conf_s* data, const long dims[DIMS], complex float* img, complex float* sens, const complex float* pattern, const complex float* mask, const complex float* TI, const complex float* b1, const complex float* kspace_data, const complex float* init)
{
	long imgs_dims[DIMS];
	long coil_dims[DIMS];
	long data_dims[DIMS];
	long pat_dims[DIMS];

	unsigned int fft_flags = FFT_FLAGS;

	if (conf->sms)
		fft_flags |= SLICE_FLAG;

	md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, imgs_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG|TIME_FLAG|TIME2_FLAG, coil_dims, dims);
	md_select_dims(DIMS, fft_flags|COIL_FLAG|TE_FLAG|MAPS_FLAG|TIME_FLAG|TIME2_FLAG, data_dims, dims);
	md_select_dims(DIMS, ~COIL_FLAG, pat_dims, data_dims);


	switch (conf->mode) {

	case MDB_T1:
        case MDB_T1_PHY:
	case MDB_T2:
        case MDB_BLOCH:

		assert(NULL == init);
		recon(conf, data, dims, imgs_dims, img, coil_dims, sens, pattern, mask, TI, b1, data_dims, kspace_data, conf->use_gpu);
		break;

	case MDB_MGRE:

		meco_recon(conf, conf->mgre_model, false, conf->fat_spec, conf->scale_fB0, true, conf->out_origin_maps, imgs_dims, img, coil_dims, sens, imgs_dims, init, mask, TI, pat_dims, pattern, data_dims, kspace_data);
		break;

	default:
		assert(0);
	}
}
