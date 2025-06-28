/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2022-2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020-2022 Martin Uecker
 * 2020 Zhengguo Tan
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/init.h"
#include "num/iovec.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "iter/italgos.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/lsqr.h"

#include "linops/fmac.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nlop_jacobian.h"
#include "nlops/someops.h"

#include "moba/meco.h"
#include "moba/exp.h"
#include "moba/T1fun.h"
#include "moba/blochfun.h"
#include "moba/moba.h"

#include "simu/signals.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

struct mobafit_bound_s {

	iter_op_data super;

	int N;
	long* dims;

	unsigned long min_flags;
	unsigned long max_flags;
	unsigned long max_norm_flags;

	float* min;
	float* max;
};

DEF_TYPEID(mobafit_bound_s)

static void mobafit_bound(iter_op_data* _data, float* dst, const float* src)
{
	assert(dst == src);

	const struct mobafit_bound_s* data = CAST_DOWN(mobafit_bound_s, _data);

	int N = data->N;

	long map_dims[N];
	long strs[N];

	md_select_dims(N, ~COEFF_FLAG, map_dims, data->dims);
	md_calc_strides(N, strs, data->dims, CFL_SIZE);

	long pos[N];
	md_set_dims(N, pos, 0);

	complex float* tmp_map = md_alloc_sameplace(N, map_dims, CFL_SIZE, dst);

	do {
		complex float* map = &MD_ACCESS(N, strs, pos, (complex float*)dst);

		if (MD_IS_SET(data->min_flags, pos[COEFF_DIM]))
			md_zsmax2(N, map_dims, strs, map, strs, map, data->min[pos[COEFF_DIM]]);

		if (MD_IS_SET(data->max_flags, pos[COEFF_DIM]))
			md_zsmin2(N, map_dims, strs, map, strs, map, data->max[pos[COEFF_DIM]]);

		if (MD_IS_SET(data->max_norm_flags, pos[COEFF_DIM])) {

			md_zabs2(N, map_dims, MD_STRIDES(N, map_dims, CFL_SIZE), tmp_map, strs, map);
			md_zdiv2(N, map_dims, strs, map, strs, map, MD_STRIDES(N, map_dims, CFL_SIZE), tmp_map);
			md_zsmin2(N, map_dims, MD_STRIDES(N, map_dims, CFL_SIZE), tmp_map, MD_STRIDES(N, map_dims, CFL_SIZE), tmp_map, data->max[pos[COEFF_DIM]]);
			md_zmul2(N, map_dims, strs, map, strs, map, MD_STRIDES(N, map_dims, CFL_SIZE), tmp_map);
		}

	} while (md_next(N, data->dims, COEFF_FLAG, pos));

	md_free(tmp_map);
}


static const char help_str[] = "Pixel-wise fitting of physical signal models.";

int main_mobafit(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* enc_file = NULL;
	const char* echo_file = NULL;
	const char* coeff_file = NULL;
	const char* b1_file = NULL;
	const char* b0_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &enc_file, "enc"),
		ARG_INFILE(true, &echo_file, "echo/contrast images"),
		ARG_OUTFILE(false, &coeff_file, "coefficients"),
	};

	float init0[DIMS] = { };
	float scale0[DIMS] = { [0 ... DIMS - 1] = 1. };

	float bound_min[DIMS] = { };
	float bound_max[DIMS] = { };

	struct mobafit_bound_s bounds;
	SET_TYPEID(mobafit_bound_s, &bounds);

	bounds.N = DIMS;
	bounds.min_flags = 0;
	bounds.max_flags = 0;
	bounds.max_norm_flags = 0;
	bounds.min = bound_min;
	bounds.max = bound_max;

	enum seq_type { /* BSSFP, FLASH, MOLLI, */ TSE, MGRE, DIFF, IR_LL, IR, SIM } seq = MGRE;

	int mgre_model = MECO_WFR2S;

	int iter = 5;

	const char* basis_file = NULL;

	bool use_magn = false;

	struct sim_data sim;
	sim.seq = simdata_seq_defaults;
	sim.voxel = simdata_voxel_defaults;
	sim.pulse = simdata_pulse_defaults;
	sim.pulse.sinc = pulse_sinc_defaults;
	sim.pulse.hs = pulse_hypsec_defaults;
	sim.grad = simdata_grad_defaults;
	sim.other = simdata_other_defaults;

	struct opt_s sim_opts[] = {

		OPTL_FLOAT(0, "ode-tol", &(sim.other.ode_tol), "", "ODE tolerance value [def: 1e-5]"),
		OPTL_FLOAT(0, "stm-tol", &(sim.other.stm_tol), "", "STM tolerance value [def: 1e-6]"),
		OPTL_SELECT(0, "ROT", enum sim_type, &(sim.seq.type), SIM_ROT,
			"homogeneously discretized simulation based on rotational matrices"),
		OPTL_SELECT(0, "ODE", enum sim_type, &(sim.seq.type), SIM_ODE,
			"full ordinary differential equation solver based simulation (default)"),
		OPTL_SELECT(0, "STM", enum sim_type, &(sim.seq.type), SIM_STM, "state-transition matrix based simulation"),
		OPTL_SELECT(0, "BLOCH", enum sim_model, &(sim.seq.model), MODEL_BLOCH, "Bloch Equations (default)"),
		OPTL_SELECT(0, "BMC", enum sim_model, &(sim.seq.model), MODEL_BMC, "Bloch-McConnell Equations"),
	};

	struct opt_s seq_opts[] = {

		/* Sequences */
		OPTL_SELECT(0, "BSSFP", enum sim_seq, &(sim.seq.seq_type), SEQ_BSSFP, "bSSFP"),
		OPTL_SELECT(0, "IR-BSSFP", enum sim_seq, &(sim.seq.seq_type), SEQ_IRBSSFP, "Inversion-Recovery bSSFP"),
		OPTL_SELECT(0, "FLASH", enum sim_seq, &(sim.seq.seq_type), SEQ_FLASH, "FLASH"),
		OPTL_SELECT(0, "IR-FLASH", enum sim_seq, &(sim.seq.seq_type), SEQ_IRFLASH, "Inversion-Recovery FLASH"),
		OPTL_SELECT(0, "STM", enum sim_type, &(sim.seq.type), SIM_STM, "state-transition matrix based simulation"),

		/* Sequences Specific Parameters */
		OPTL_FLOAT(0, "TR", &(sim.seq.tr), "float", "Repetition time [s]"),
		OPTL_FLOAT(0, "TE", &(sim.seq.te), "float", "Echo time [s]"),
		OPTL_PINT(0, "Nspins", &(sim.seq.spin_num), "int", "Number of averaged spins"),
		OPTL_PINT(0, "Nrep", &(sim.seq.rep_num), "int", "Number of repetitions"),
		OPTL_SET(0, "pinv", &(sim.seq.perfect_inversion), "Use perfect inversions"),
		OPTL_FLOAT(0, "ipl", &(sim.seq.inversion_pulse_length), "float", "Inversion Pulse Length [s]"),
		OPTL_FLOAT(0, "isp", &(sim.seq.inversion_spoiler), "float", "Inversion Spoiler Gradient Length [s]"),
		OPTL_FLOAT(0, "ppl", &(sim.seq.prep_pulse_length), "float", "Preparation Pulse Length [s]"),
		OPTL_PINT(0, "av-spokes", &(sim.seq.averaged_spokes), "", "Number of averaged consecutive spokes"),
		OPTL_FLOAT(0, "m0", &(sim.voxel.m0[0]), "float", "m0"),

		/* Pulse Specific Parameters */
		OPTL_FLOAT(0, "Trf", &(sim.pulse.rf_end), "float", "Pulse Duration [s]"), /* Assumes to start at t=0 */
		OPTL_FLOAT(0, "FA", &(sim.pulse.sinc.super.flipangle), "float", "Flipangle [deg]"),
		OPTL_FLOAT(0, "BWTP", &(sim.pulse.sinc.bwtp), "float", "Bandwidth-Time-Product"),

		/* Voxel Specific Parameters */
		OPTL_FLOAT(0, "off", &(sim.voxel.w), "float", "Off-Resonance [rad/s]"),

		/* Slice Profile Parameters */
		OPTL_FLOAT(0, "sl-grad", &(sim.grad.sl_gradient_strength), "float", "Strength of slice-selection gradient [T/m]"),
		OPTL_FLOAT(0, "slice-thickness", &(sim.seq.slice_thickness), "float", "Thickness of simulated slice. [m]"),
		OPTL_FLOAT(0, "nom-slice-thickness", &(sim.seq.nom_slice_thickness), "float", "Nominal thickness of simulated slice. [m]"),
	};

	struct opt_s other_opts[] = {

		OPTL_FLOAT(0, "ode-tol", &(sim.other.ode_tol), "", "ODE tolerance value [def: 1e-5]"),
		OPTL_FLOAT(0, "stm-tol", &(sim.other.stm_tol), "", "STM tolerance value [def: 1e-6]"),
		OPTL_FLOAT(0, "sampling-rate", &(sim.other.sampling_rate), "", "Sampling rate of RF pulse used for ROT simulation in Hz [def: 1e6 Hz]"),
	};

	struct opt_s pool_opts[] = {

		OPT_PINT('P', &(sim.voxel.P), "int", "Number of pools"),
	};

	struct opt_s cest_opts[] = {

		OPTL_FLOAT(0, "b1", &(sim.cest.b1_amp), "float", "B1 amplitude [mu T]"),
		OPTL_FLOAT(0, "b0", &(sim.cest.b0), "float", "B0 [T]"),
		OPTL_FLOAT(0, "gamma", &(sim.cest.gamma), "float", "Gyromagnetic ratio [Mhz/T]"),
		OPTL_FLOAT(0, "max", &(sim.cest.off_start), "float", "Max offset [ppm]"),
		OPTL_FLOAT(0, "min", &(sim.cest.off_stop), "float", "Min offset [ppm]"),
		OPTL_PINT(0, "n_p", &(sim.cest.n_pulses), "int", "Number of pulses"),
		OPTL_FLOAT(0, "t_d", &(sim.cest.t_d), "float", "Interpulse delay [s]"),
		OPTL_FLOAT(0, "t_pp", &(sim.cest.t_pp), "float", "Post-preparation delay [s]"),
		OPTL_SET(0, "ref_scan", &(sim.cest.ref_scan), "Use reference scan"),
		OPTL_FLOAT(0, "ref_scan_ppm", &(sim.cest.ref_scan_ppm), "float", "Offset for ref. scan [ppm]"),
	};

	const struct opt_s opts[] = {

#if 0
		OPT_SELECT('F', enum seq_type, &seq, FLASH, "FLASH"),
		OPT_SELECT('B', enum seq_type, &seq, BSSFP, "bSSFP"),
		OPT_SELECT('M', enum seq_type, &seq, MOLLI, "MOLLI"),
#endif
		OPT_SELECT('T', enum seq_type, &seq, TSE, "Multi-Echo Spin Echo: f(M0, R2) = M0 * exp(-t * R2)"),
		OPT_SELECT('I', enum seq_type, &seq, IR, "Inversion Recovery: f(M0, R1, c) =  M0 * (1 - exp(-t * R1 + c))"),
		OPT_SELECT('L', enum seq_type, &seq, IR_LL, "Inversion Recovery Look-Locker"),
		OPT_SELECT('G', enum seq_type, &seq, MGRE, "MGRE"),
		OPT_SELECT('D', enum seq_type, &seq, DIFF, "diffusion"),
		OPT_SELECT('S', enum seq_type, &seq, SIM, "Simulation based fitting"),
		OPT_PINT('m', &mgre_model, "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF } [default: WFR2S]"),
		OPT_SET('a', &use_magn, "fit magnitude of signal model to data"),
		OPT_PINT('i', &iter, "iter", "Number of IRGNM steps"),
		OPT_SET('g', &bart_use_gpu, "use gpu"),
		OPT_INFILE('B', &basis_file, "file", "temporal (or other) basis"),
		OPTL_FLVECN(0, "init", init0, "Initial values of parameters in model-based reconstruction"),
		OPTL_FLVECN(0, "scale", scale0, "Scaling"),

		OPTL_ULONG(0, "min-flag", &(bounds.min_flags), "flags", "Apply minimum constraint on selected maps"),
		OPTL_ULONG(0, "max-flag", &(bounds.max_flags), "flags", "Apply maximum constraint on selected maps"),
		OPTL_ULONG(0, "max-mag-flag", &(bounds.max_norm_flags), "flags", "Apply maximum magnitude constraint on selected maps"),
		OPTL_FLVECN(0, "min", bound_min, "Min bound (map must be selected with \"min-flag\")"),
		OPTL_FLVECN(0, "max", bound_max, "Max bound (map must be selected with \"max-flag\" or \"max-mag-flag\")"),
		OPTL_INFILE(0, "b1map", &b1_file, "[deg]", "Input B1 map as cfl file"),
		OPTL_INFILE(0, "b0map", &b0_file, "[rad/s]", "Input B0 map as cfl file"),
		OPTL_SUBOPT(0, "seq", "...", "configure sequence parameters for simulation based fitting", ARRAY_SIZE(seq_opts), seq_opts),
		OPTL_SUBOPT(0, "sim", "...", "configure simulation parameters", ARRAY_SIZE(sim_opts), sim_opts),
		OPTL_SUBOPT(0, "other", "...", "configure other simulation parameters", ARRAY_SIZE(other_opts), other_opts),
		OPTL_SUBOPT(0, "pool", "...", "configure pool parameters for BMC simulation", ARRAY_SIZE(pool_opts), pool_opts),
		OPTL_SUBOPT(0, "cest", "...", "configure parameters for CEST simulation", ARRAY_SIZE(cest_opts), cest_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init_gpu_support();

	long bas_dims[DIMS];
	complex float* basis = NULL;

	if (NULL != basis_file) {

		basis = load_cfl(basis_file, DIMS, bas_dims);
		md_zconj(DIMS, bas_dims, basis, basis);
	}

	long enc_dims[DIMS];
	complex float* enc = load_cfl(enc_file, DIMS, enc_dims);

	long y_dims[DIMS];
	complex float* y = load_cfl(echo_file, DIMS, y_dims);

	long y_sig_dims[DIMS];
	md_copy_dims(DIMS, y_sig_dims, y_dims);

	if (NULL == basis) {

		assert(y_dims[TE_DIM] == enc_dims[TE_DIM]);

	} else {

		assert(bas_dims[TE_DIM] == enc_dims[TE_DIM]);

		if (1 != y_dims[TE_DIM]) {

			assert(y_dims[TE_DIM] == enc_dims[TE_DIM]);

			y_dims[COEFF_DIM] = bas_dims[COEFF_DIM];
			y_dims[TE_DIM] = 1;

			complex float* ny = anon_cfl(NULL, DIMS, y_dims);

			md_ztenmul(DIMS, y_dims, ny, bas_dims, basis, y_sig_dims, y);

			unmap_cfl(DIMS, y_sig_dims, y);

			y = ny;

		} else {

			y_sig_dims[TE_DIM] = bas_dims[TE_DIM];
			y_sig_dims[COEFF_DIM] = 1;

			assert(y_dims[COEFF_DIM] == bas_dims[COEFF_DIM]);
		}
	}

	long x_dims[DIMS];
	md_select_dims(DIMS, ~(TE_FLAG | COEFF_FLAG), x_dims, y_dims);

	switch (seq) {

	case IR:
	case IR_LL:

		x_dims[COEFF_DIM] = 3;
		break;

	case MGRE:

	//	assert(1 == enc_dims[TE_DIM]);

		x_dims[COEFF_DIM] = get_num_of_coeff(mgre_model);
		break;

	case TSE:

		assert(1 == enc_dims[COEFF_DIM]);

		md_zsmul(DIMS, enc_dims, enc, enc, -1.);

		x_dims[COEFF_DIM] = 2;
		break;

	case DIFF:

		x_dims[COEFF_DIM] = enc_dims[COEFF_DIM] + 1;
		break;

	case SIM:

		x_dims[COEFF_DIM] = (1 == sim.voxel.P) ? 4 : (5 * sim.voxel.P) - 1;
		break;

	default:

		error("sequence type not supported\n");
	}


	complex float* x = create_cfl(coeff_file, DIMS, x_dims);

	md_zfill(DIMS, x_dims, x, 1.);


	long y_patch_dims[DIMS];
	long x_patch_dims[DIMS];
	long y_patch_sig_dims[DIMS];

	md_select_dims(DIMS, FFT_FLAGS | TE_FLAG | COEFF_FLAG, y_patch_dims, y_dims);
	md_select_dims(DIMS, FFT_FLAGS | TE_FLAG | COEFF_FLAG, y_patch_sig_dims, y_sig_dims);
	md_select_dims(DIMS, FFT_FLAGS | TE_FLAG | COEFF_FLAG, x_patch_dims, x_dims);


	// create signal model
	const struct nlop_s* nlop = NULL;
	struct moba_conf_s *moba_conf;
	moba_conf = xmalloc(sizeof(struct moba_conf_s));

	switch (seq) {

		long dims[DIMS];

	case IR:

		assert(md_check_equal_dims(DIMS, y_patch_dims, y_patch_sig_dims, ~0UL));

		md_copy_dims(DIMS, dims, y_patch_dims);
		dims[COEFF_DIM] = enc_dims[COEFF_DIM];

		nlop = nlop_ir_create(DIMS, dims, enc);
		break;

	case IR_LL:

		long map_dims[DIMS];
		md_select_dims(DIMS, ~(COEFF_FLAG | TE_FLAG), map_dims, x_patch_dims);

		nlop = nlop_T1_create(DIMS, map_dims, y_patch_sig_dims, x_patch_dims, enc_dims, enc, 1, 1);

		if (NULL != basis) {

			long max_dims[DIMS];
			md_max_dims(DIMS, ~0UL, max_dims, bas_dims, y_patch_sig_dims);

			unsigned long oflags = ~md_nontriv_dims(DIMS, y_patch_dims);
			unsigned long iflags = ~md_nontriv_dims(DIMS, y_patch_sig_dims);
			unsigned long bflags = ~md_nontriv_dims(DIMS, bas_dims);

			const struct nlop_s* nlop_bas = nlop_from_linop_F(linop_fmac_create(DIMS, max_dims, oflags, iflags, bflags, basis));
			nlop = nlop_chain_FF(nlop, nlop_bas);

			long tdims[DIMS];
			md_transpose_dims(DIMS, 5, 6, tdims, y_patch_dims);
			nlop = nlop_reshape_out_F(nlop, 0, DIMS, tdims);
			nlop = nlop_zrprecomp_jacobian_F(nlop);
			nlop = nlop_reshape_out_F(nlop, 0, DIMS, y_patch_dims);

			auto tmp = linop_stack_FF(6, 6, linop_identity_create(DIMS, map_dims), linop_identity_create(DIMS, map_dims));
			tmp = linop_stack_FF(6, 6, tmp, linop_zreal_create(DIMS, map_dims));
			nlop = nlop_chain_FF(nlop_from_linop_F(tmp), nlop);
		}

		break;

	case MGRE:

		static float scale_fB0[2] = { 0., 1. };
		assert(md_check_equal_dims(DIMS, y_patch_dims, y_patch_sig_dims, ~0UL));

		nlop = nlop_meco_create(DIMS, y_patch_dims, x_patch_dims, enc, mgre_model, false, FAT_SPEC_1, scale_fB0);
		break;

	case TSE:
	case DIFF:

		assert(md_check_equal_dims(DIMS, y_patch_dims, y_patch_sig_dims, ~0UL));

		md_copy_dims(DIMS, dims, y_patch_dims);
		dims[COEFF_DIM] = enc_dims[COEFF_DIM];

		{
			auto nl = nlop_exp_create(DIMS, dims, enc);
			nlop = nlop_flatten(nl);
			nlop_free(nl);
		}
		break;

	case SIM:

		md_select_dims(DIMS, ~(COEFF_FLAG | TE_FLAG), map_dims, x_patch_dims);

		const complex float *b1 = NULL;
		long b1_dims[DIMS];

		const complex float *b0 = NULL;
		long b0_dims[DIMS];

		long bloch_dims[DIMS];
		long der_dims[DIMS];
		long out_dims[DIMS];
		long in_dims[DIMS];

		moba_conf->model = MDB_BLOCH;
		moba_conf->sim = sim;
		moba_conf->other = moba_other_defaults;

		for (int i = 0; i < x_dims[COEFF_DIM]; i++) {

			moba_conf->other.initval[i] = init0[i];
			moba_conf->other.scale[i] = scale0[i];
		}

		md_copy_dims(DIMS, bloch_dims, x_dims);
		bloch_dims[TE_DIM] = y_patch_dims[TE_DIM];
		bloch_dims[READ_DIM] = x_patch_dims[READ_DIM];
		bloch_dims[PHS1_DIM] = y_patch_dims[PHS1_DIM];

		if (NULL != b1_file)
			b1 = load_cfl(b1_file, DIMS, b1_dims);

		if (NULL != b0_file)
			b0 = load_cfl(b0_file, DIMS, b0_dims);

		md_select_dims(DIMS, FFT_FLAGS | TE_FLAG | COEFF_FLAG | TIME2_FLAG, der_dims, bloch_dims);
		md_select_dims(DIMS, FFT_FLAGS | TIME_FLAG | TIME2_FLAG, map_dims, bloch_dims);
		md_select_dims(DIMS, FFT_FLAGS | TE_FLAG | TIME_FLAG | TIME2_FLAG, out_dims, bloch_dims);
		md_select_dims(DIMS, FFT_FLAGS | COEFF_FLAG | TIME_FLAG | TIME2_FLAG, in_dims, bloch_dims);

		moba_conf->sim.seq.rep_num = y_dims[TE_DIM];

		nlop = nlop_bloch_create(DIMS, der_dims, map_dims, out_dims, in_dims, b1, b0, moba_conf);
		break;
	}

	if (use_magn) {

		assert(NULL == basis);
		nlop = nlop_chain_FF(nlop, nlop_zabs_create(DIMS, y_patch_dims));
	}

	assert(nlop);


	complex float init[DIMS] = { [0 ... DIMS - 1] = 1. };
	complex float scale[DIMS] = { [0 ... DIMS - 1] = 1. };

	for (long i = 0; i < x_dims[COEFF_DIM]; i++) {

		init[i] = init0[i];
		scale[i] = scale0[i];

		bound_max[i] /= (scale0[i] ?: 1);
		bound_min[i] /= (scale0[i] ?: 1);
	}

	long c_dims[DIMS];
	md_select_dims(DIMS, COEFF_FLAG, c_dims, x_dims);

	long c_strs[DIMS];
	long x_strs[DIMS];

	md_calc_strides(DIMS, c_strs, c_dims, CFL_SIZE);
	md_calc_strides(DIMS, x_strs, x_dims, CFL_SIZE);

	md_zfill(DIMS, x_dims, x, 1.);
	md_zmul2(DIMS, x_dims, x_strs, x, x_strs, x, c_strs, init);
	md_zdiv2(DIMS, x_dims, x_strs, x, x_strs, x, c_strs, scale);

	bounds.dims = x_patch_dims;

	for (long i = 0; i < x_dims[COEFF_DIM]; i++) {

		if (1. != scale[i]) {

			auto lop_scale = linop_cdiag_create(DIMS, x_patch_dims, COEFF_FLAG, scale);

			nlop = nlop_chain_FF(nlop_from_linop_F(lop_scale), nlop);
			break;
		}
	}


	struct iter_conjgrad_conf conjgrad_conf = iter_conjgrad_defaults;
	conjgrad_conf.Bi = md_calc_size(3, x_patch_dims);

	struct lsqr_conf lsqr_conf = lsqr_defaults;
	lsqr_conf.it_gpu = false;

	const struct operator_p_s* lsqr = lsqr2_create(&lsqr_conf, iter2_conjgrad, CAST_UP(&conjgrad_conf), NULL, &nlop->derivative[0][0], NULL, 0, NULL, NULL, NULL);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	irgnm_conf.iter = iter;


	complex float* y_patch = NULL;
	complex float* x_patch = NULL;

	if (bart_use_gpu) {

#ifdef USE_CUDA
		y_patch = md_alloc_gpu(DIMS, y_patch_dims, CFL_SIZE);
		x_patch = md_alloc_gpu(DIMS, x_patch_dims, CFL_SIZE);
#else
		error("Compiled without GPU support!\n");
#endif

	} else {

		y_patch = md_alloc(DIMS, y_patch_dims, CFL_SIZE);
		x_patch = md_alloc(DIMS, x_patch_dims, CFL_SIZE);
	}


	long pos[DIMS] = { };

	do {
		md_copy_block(DIMS, pos, y_patch_dims, y_patch, y_dims, y, CFL_SIZE);
		md_copy_block(DIMS, pos, x_patch_dims, x_patch, x_dims, x, CFL_SIZE);

		if (0. == md_znorm(DIMS, y_patch_dims, y_patch)) {

			md_zfill(DIMS, x_patch_dims, x_patch, 0.);
			continue;
		}

		iter4_irgnm2(CAST_UP(&irgnm_conf), nlop,
				2 * md_calc_size(DIMS, x_patch_dims), (float*)x_patch, NULL,
				2 * md_calc_size(DIMS, y_patch_dims), (const float*)y_patch, lsqr,
				(struct iter_op_s){ mobafit_bound, CAST_UP(&bounds) });

		md_copy_block(DIMS, pos, x_dims, x, x_patch_dims, x_patch, CFL_SIZE);

	} while (md_next(DIMS, y_dims, ~(FFT_FLAGS | TE_FLAG | COEFF_FLAG), pos));


	md_free(x_patch);
	md_free(y_patch);


	operator_p_free(lsqr);
	nlop_free(nlop);
	xfree(moba_conf);

	md_zmul2(DIMS, x_dims, x_strs, x, x_strs, x, c_strs, scale);

	unmap_cfl(DIMS, y_dims, y);
	unmap_cfl(DIMS, enc_dims, enc);
	unmap_cfl(DIMS, x_dims, x);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);

	return 0;
}

