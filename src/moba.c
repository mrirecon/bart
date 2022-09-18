/* Copyright 2013. The Regents of the University of California.
 * Copyright 2019-2021. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2021-2022. Institute of Medical Engineering. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker, Nick Scholand
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/filter.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"
#include "misc/version.h"

#include "simu/pulse.h"
#include "simu/simulation.h"

#include "noncart/nufft.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/iter2.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "moba/optreg.h"
#include "moba/recon.h"
#include "moba/moba.h"
#include "moba/meco.h"

static const char help_str[] = "Model-based nonlinear inverse reconstruction";


static void edge_filter1(const long map_dims[DIMS], complex float* dst, float lambda)
{
	klaplace(DIMS, map_dims, READ_FLAG|PHS1_FLAG, dst);
	md_zreal(DIMS, map_dims, dst, dst);
	md_zsqrt(DIMS, map_dims, dst, dst);

	md_zsmul(DIMS, map_dims, dst, dst, -2.);
	md_zsadd(DIMS, map_dims, dst, dst, 1.);
	md_zatanr(DIMS, map_dims, dst, dst);

	md_zsmul(DIMS, map_dims, dst, dst, -1. / M_PI);
	md_zsadd(DIMS, map_dims, dst, dst, 1.);
	md_zsmul(DIMS, map_dims, dst, dst, lambda);
}

static void edge_filter2(const long map_dims[DIMS], complex float* dst, float lambda)
{
	float beta = 100.;

	klaplace(DIMS, map_dims, READ_FLAG|PHS1_FLAG, dst);
	md_zspow(DIMS, map_dims, dst, dst, 0.5);

	md_zsmul(DIMS, map_dims, dst, dst, -beta * 2.);
	md_zsadd(DIMS, map_dims, dst, dst, beta);

	md_zatanr(DIMS, map_dims, dst, dst);
	md_zsmul(DIMS, map_dims, dst, dst, -(50. * lambda) / M_PI);
	md_zsadd(DIMS, map_dims, dst, dst, 25. * lambda);
}


int main_moba(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* ksp_file = NULL;
	const char* TI_file = NULL;
	const char* out_file = NULL;
	const char* sens_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_INFILE(true, &TI_file, "TI/TE"),
		ARG_OUTFILE(true, &out_file, "output"),
		ARG_OUTFILE(false, &sens_file, "sensitivities"),
	};

	float restrict_fov = -1.;
	float oversampling = 1.f;

	float kfilter_strength = 2e-3;

	bool normalize_scaling = false;
	float scaling = 1.;
	float scaling_psf = 1.;

	const char* psf_file = NULL;
	const char* traj_file = NULL;
	const char* init_file = NULL;
        const char* input_b1 = NULL;

	struct moba_conf conf = moba_defaults;
	struct opt_reg_s ropts;
	conf.ropts = &ropts;

	long img_vec[3] = { 0 };

        struct moba_conf_s data;

        data.sim.seq = simdata_seq_defaults;
        data.sim.voxel = simdata_voxel_defaults;
        data.sim.pulse = simdata_pulse_defaults;
        data.sim.pulse.hs = hs_pulse_defaults;
        data.sim.grad = simdata_grad_defaults;
        data.sim.tmp = simdata_tmp_defaults;
	data.sim.other = simdata_other_defaults;
        data.other = moba_other_defaults;

        // FIXME: Move to separate function to reuse it for sim.c
        struct opt_s seq_opts[] = {

                /* Sequence Specific Parameters */
		OPTL_SELECT(0, "BSSFP", enum sim_seq, &(data.sim.seq.seq_type), SEQ_BSSFP, "bSSFP"),
		OPTL_SELECT(0, "IR-BSSFP", enum sim_seq, &(data.sim.seq.seq_type), SEQ_IRBSSFP, "Inversion-Recovery bSSFP"),
		OPTL_SELECT(0, "FLASH", enum sim_seq, &(data.sim.seq.seq_type), SEQ_FLASH, "FLASH"),
		OPTL_SELECT(0, "IR-FLASH", enum sim_seq, &(data.sim.seq.seq_type), SEQ_IRFLASH, "Inversion-Recovery FLASH"),
		OPTL_FLOAT(0, "TR", &(data.sim.seq.tr), "float", "Repetition time [s]"),
		OPTL_FLOAT(0, "TE", &(data.sim.seq.te), "float", "Echo time [s]"),
		OPTL_INT(0, "Nspins", &(data.sim.seq.spin_num), "int", "Number of averaged spins"),
		OPTL_INT(0, "Nrep", &(data.sim.seq.rep_num), "int", "Number of repetitions"),
		OPTL_SET(0, "pinv", &(data.sim.seq.perfect_inversion), "Use perfect inversions"),
		OPTL_FLOAT(0, "ipl", &(data.sim.seq.inversion_pulse_length), "float", "Inversion Pulse Length [s]"),
		OPTL_FLOAT(0, "isp", &(data.sim.seq.inversion_spoiler), "float", "Inversion Spoiler Gradient Length [s]"),
		OPTL_FLOAT(0, "ppl", &(data.sim.seq.prep_pulse_length), "float", "Preparation Pulse Length [s]"),
		OPTL_INT(0, "av-spokes", &(data.sim.seq.averaged_spokes), "", "Number of averaged consecutive spokes"),

        	/* Pulse Specific Parameters */
		OPTL_FLOAT(0, "Trf", &(data.sim.pulse.rf_end), "float", "Pulse Duration [s]"), /* Assumes to start at t=0 */
		OPTL_FLOAT(0, "FA", &(data.sim.pulse.flipangle), "float", "Flipangle [deg]"),
		OPTL_FLOAT(0, "BWTP", &(data.sim.pulse.bwtp), "float", "Bandwidth-Time-Product"),

                /* Voxel Specific Parameters */
                OPTL_FLOAT(0, "off", &(data.sim.voxel.w), "float", "Off-Resonance [rad/s]"),

		/* Slice Profile Parameters */
                OPTL_FLOAT(0, "sl-grad", &(data.sim.grad.sl_gradient_strength), "float", "Strength of slice-selection gradient [T/m]"),
                OPTL_FLOAT(0, "slice-thickness", &(data.sim.seq.slice_thickness), "float", "Thickness of simulated slice. [m]"),
		OPTL_FLOAT(0, "nom-slice-thickness", &(data.sim.seq.nom_slice_thickness), "float", "Nominal thickness of simulated slice. [m]"),

        };

        struct opt_s sim_opts[] = {

                OPTL_SELECT(0, "ODE", enum sim_type, &(data.sim.seq.type), SIM_ODE, "full ordinary differential equation solver based simulation"),
                OPTL_SELECT(0, "STM", enum sim_type, &(data.sim.seq.type), SIM_STM, "state-transition matrix based simulation (default)"),
        };


	int tvscales_N = 4;
	float tvscales[4] = { 0. };

        struct opt_s other_opts[] = {

		// FIXME: MGRE can have 5 parameters
                OPTL_FLVEC4(0, "pscale", &(data.other.scale), "s1:s2:s3:s4", "Scaling of parameters in model-based reconstruction"),
                OPTL_FLVEC4(0, "pinit", &(data.other.initval), "i1:i2:i3:i4", "Initial values of parameters in model-based reconstruction"),
                OPTL_INFILE(0, "b1map", &input_b1, "[deg]", "Input B1 map as cfl file"),
                OPTL_FLVEC4(0, "tvscale", &tvscales, "s1:s2:s3:s4", "Scaling of derivatives in TV penalty"),
        };

	opt_reg_init(&ropts);

	const struct opt_s opts[] = {

                //FIXME: Sort options into optimization and others interface
		{ 'r', NULL, true, OPT_SPECIAL, opt_reg_moba, &ropts, "<T>:A:B:C", "generalized regularization options (-rh for help)" },
		OPT_SELECT('L', enum mdb_t, &conf.mode, MDB_T1, "T1 mapping using model-based look-locker"),
                OPT_SELECT('P', enum mdb_t, &conf.mode, MDB_T1_PHY, "T1 mapping using reparameterized (M0, R1, alpha) model-based look-locker (TR required!)"),
		OPT_SELECT('F', enum mdb_t, &conf.mode, MDB_T2, "T2 mapping using model-based Fast Spin Echo"),
		OPT_SELECT('G', enum mdb_t, &conf.mode, MDB_MGRE, "T2* mapping using model-based multiple gradient echo"),
                OPTL_SELECT(0, "bloch", enum mdb_t, &conf.mode, MDB_BLOCH, "Bloch model-based reconstruction"),
		OPT_UINT('m', &conf.mgre_model, "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF } [default: WFR2S]"),
		OPT_UINT('l', &conf.opt_reg, "\b1/-l2", "  toggle l1-wavelet or l2 regularization."), // extra spaces needed because of backsapce \b earlier
		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPTL_FLOAT('R', "reduction", &conf.redu, "redu", "reduction factor"),
		OPT_FLOAT('T', &conf.damping, "damp", "damping on temporal frames"),
		OPT_FLOAT('j', &conf.alpha_min, "minreg", "Minimum regularization parameter"),
		OPT_FLOAT('u', &conf.rho, "rho", "ADMM rho [default: 0.01]"),
		OPT_UINT('C', &conf.inner_iter, "iter", "inner iterations"),
		OPT_FLOAT('s', &conf.step, "step", "step size"),
		OPT_FLOAT('B', &conf.lower_bound, "bound", "lower bound for relaxation"),
		OPT_FLVEC2('b', &conf.scale_fB0, "SMO:SC", "B0 field: spatial smooth level; scaling [default: 222.; 1.]"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_SET('N', &conf.auto_norm, "(normalize)"),
		OPT_FLOAT('f', &restrict_fov, "FOV", ""),
		OPT_INFILE('p', &psf_file, "PSF", ""),
		OPT_SET('J', &conf.stack_frames, "Stack frames for joint recon"),
		OPT_SET('M', &conf.sms, "Simultaneous Multi-Slice reconstruction"),
		OPT_SET('O', &conf.out_origin_maps, "(Output original maps from reconstruction without post processing)"),
		OPT_SET('g', &conf.use_gpu, "use gpu"),
		OPTL_INT(0, "multi-gpu", &conf.num_gpu, "num", "number of gpus to use"),
		OPT_INFILE('I', &init_file, "init", "File for initialization"),
		OPT_INFILE('t', &traj_file, "traj", "K-space trajectory"),
		OPT_FLOAT('o', &oversampling, "os", "Oversampling factor for gridding [default: 1.]"),
		OPTL_VEC3(0, "img_dims", &img_vec, "x:y:z", "dimensions"),
		OPT_SET('k', &conf.k_filter, "k-space edge filter for non-Cartesian trajectories"),
		OPTL_SELECT(0, "kfilter-1", enum edge_filter_t, &conf.k_filter_type, EF1, "k-space edge filter 1"),
		OPTL_SELECT(0, "kfilter-2", enum edge_filter_t, &conf.k_filter_type, EF2, "k-space edge filter 2"),
		OPT_FLOAT('e', &kfilter_strength, "kfilter_strength", "strength for k-space edge filter [default: 2e-3]"),
		OPT_CLEAR('n', &conf.auto_norm, "(disable normalization of parameter maps for thresholding)"),
		OPTL_CLEAR(0, "no_alpha_min_exp_decay", &conf.alpha_min_exp_decay, "(Use hard minimum instead of exponential decay towards alpha_min)"),
		OPTL_FLOAT(0, "sobolev_a", &conf.sobolev_a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPTL_FLOAT(0, "sobolev_b", &conf.sobolev_b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPTL_SELECT(0, "fat_spec_0", enum fat_spec, &conf.fat_spec, FAT_SPEC_0, "select fat spectrum from ISMRM fat-water tool"),
		OPTL_FLOAT(0, "scale_data", &scaling, "", "scaling factor for data"),
		OPTL_FLOAT(0, "scale_psf", &scaling_psf, "", "(scaling factor for PSF)"),
		OPTL_SET(0, "normalize_scaling", &normalize_scaling, "(normalize scaling by data / PSF)"),
                OPTL_SUBOPT(0, "seq", "...", "configure sequence parameters", ARRAY_SIZE(seq_opts), seq_opts),
                OPTL_SUBOPT(0, "sim", "...", "configure simulation parameters", ARRAY_SIZE(sim_opts), sim_opts),
                OPTL_SUBOPT(0, "other", "...", "configure other parameters", ARRAY_SIZE(other_opts), other_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (conf.use_gpu || (0 < conf.num_gpu)) {

		num_init_multigpu(MAX(1, conf.num_gpu));

	} else {

		num_init();
	}
	
#ifdef USE_CUDA
	cuda_use_global_memory();
#endif

        data.model = conf.mode;

        if (MDB_T1_PHY == conf.mode)
                debug_printf(DP_INFO, "The TR for MDB_T1_PHY is %f s!\n", data.sim.seq.tr);

        // debug_sim(&(data.sim));
        // debug_other(&(data.other));
	if (use_compat_to_version("v0.6.00"))
		conf.scaling_M0 = 2.;


	if (conf.ropts->r > 0)
		conf.algo = ALGO_ADMM;

	while ((0 < tvscales_N) && (0. == tvscales[tvscales_N - 1]))
		tvscales_N--;

	data.other.tvscales_N = tvscales_N;

	for (int i = 0; i < tvscales_N; i++)
		data.other.tvscales[i] = tvscales[i];



	long ksp_dims[DIMS];
	complex float* kspace_data = load_cfl(ksp_file, DIMS, ksp_dims);

	long TI_dims[DIMS];
	const complex float* TI = load_cfl(TI_file, DIMS, TI_dims);

	assert(TI_dims[TE_DIM] == ksp_dims[TE_DIM]);
	assert(1 == ksp_dims[MAPS_DIM]);

	if (conf.sms) {

		debug_printf(DP_INFO, "SMS Model-based reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		fftmod(DIMS, ksp_dims, SLICE_FLAG, kspace_data, kspace_data); // fftmod to get correct slice order in output
	}

	long grid_dims[DIMS];
	md_copy_dims(DIMS, grid_dims, ksp_dims);

	long traj_dims[DIMS];
	long traj_strs[DIMS];
	complex float* traj = NULL;

	long img_dims[DIMS];

	if (NULL != traj_file) {

		traj = load_cfl(traj_file, DIMS, traj_dims);

		md_calc_strides(DIMS, traj_strs, traj_dims, CFL_SIZE);

		md_zsmul(DIMS, traj_dims, traj, traj, oversampling);

		if (0 == md_calc_size(3, img_vec)) {

			estimate_im_dims(DIMS, FFT_FLAGS, img_dims, traj_dims, traj);
			md_copy_dims(3, img_vec, img_dims);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", img_vec[0], img_vec[1], img_vec[2]);
		}


		if (!use_compat_to_version("v0.7.00")) {

			md_zsmul(DIMS, traj_dims, traj, traj, 2.);

			NESTED(long, dbl, (long x)) { return (x > 1) ? (2 * x) : 1; };

			grid_dims[READ_DIM] = dbl(img_vec[0]);
			grid_dims[PHS1_DIM] = dbl(img_vec[1]);
			grid_dims[PHS2_DIM] = dbl(img_vec[2]);

		} else {

			long grid_size = ksp_dims[1] * oversampling;
			grid_dims[READ_DIM] = grid_size;
			grid_dims[PHS1_DIM] = grid_size;
			grid_dims[PHS2_DIM] = 1L;
		}

		if (-1 == restrict_fov)
			restrict_fov = 0.5;

		conf.noncartesian = true;
	}

	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|COEFF_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, img_dims, grid_dims);


	img_dims[COEFF_DIM] = moba_get_nr_of_coeffs(&conf, grid_dims[TE_DIM]); // grid_dims[TE_DIM] is only used for MECO_PI == conf.mgre_model


	long img_strs[DIMS];
	md_calc_strides(DIMS, img_strs, img_dims, CFL_SIZE);

	long single_map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, single_map_dims, grid_dims);

	long single_map_strs[DIMS];
	md_calc_strides(DIMS, single_map_strs, single_map_dims, CFL_SIZE);

	long coil_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS|COIL_FLAG|MAPS_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, coil_dims, grid_dims);

	long coil_strs[DIMS];
	md_calc_strides(DIMS, coil_strs, coil_dims, CFL_SIZE);

	complex float* img = create_cfl(out_file, DIMS, img_dims);
	complex float* single_map = anon_cfl("", DIMS, single_map_dims);

	long dims[DIMS];
	md_copy_dims(DIMS, dims, grid_dims);

	dims[COEFF_DIM] = img_dims[COEFF_DIM];

	long msk_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, msk_dims, grid_dims);

	long msk_strs[DIMS];
	md_calc_strides(DIMS, msk_strs, msk_dims, CFL_SIZE);

	complex float* mask = NULL;
	bool sensout = (NULL != sens_file);
	complex float* sens = (sensout ? create_cfl : anon_cfl)(sensout ? sens_file : "", DIMS, coil_dims);


	md_zfill(DIMS, img_dims, img, 1.0);
	md_clear(DIMS, coil_dims, sens, CFL_SIZE);

	complex float* k_grid_data = anon_cfl("", DIMS, grid_dims);

	complex float* pattern = NULL;
	long pat_dims[DIMS];
	

	if (NULL != psf_file) {

		complex float* tmp_psf = load_cfl(psf_file, DIMS, pat_dims);

		pattern = anon_cfl("", DIMS, pat_dims);

		md_copy(DIMS, pat_dims, pattern, tmp_psf, CFL_SIZE);

		unmap_cfl(DIMS, pat_dims, tmp_psf);

		md_copy(DIMS, grid_dims, k_grid_data, kspace_data, CFL_SIZE);

		unmap_cfl(DIMS, ksp_dims, kspace_data);

		if (!md_check_compat(DIMS, COIL_FLAG, ksp_dims, pat_dims))
			error("pattern not compatible with kspace dimensions\n");

		if (-1 == restrict_fov)
			restrict_fov = 0.5;

		conf.noncartesian = true;

	} else if (NULL != traj_file) {

		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = false;

		struct linop_s* nufft_op_k = NULL;

		md_select_dims(DIMS, FFT_FLAGS|TE_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, pat_dims, grid_dims);

		pattern = anon_cfl("", DIMS, pat_dims);

		// Gridding sampling pattern
		
		complex float* psf = NULL;

		long wgh_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, wgh_dims, ksp_dims);

		complex float* wgh = md_alloc(DIMS, wgh_dims, CFL_SIZE);

		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, wgh, kspace_data);

		psf = compute_psf(DIMS, pat_dims, traj_dims, traj, traj_dims, NULL, wgh_dims, wgh, false, false);

		fftuc(DIMS, pat_dims, FFT_FLAGS, pattern, psf);

		md_free(wgh);
		md_free(psf);

		// Gridding raw data

		nufft_op_k = nufft_create(DIMS, ksp_dims, grid_dims, traj_dims, traj, NULL, nufft_conf);
		linop_adjoint(nufft_op_k, DIMS, grid_dims, k_grid_data, DIMS, ksp_dims, kspace_data);
		fftuc(DIMS, grid_dims, FFT_FLAGS, k_grid_data, k_grid_data);

		linop_free(nufft_op_k);

		unmap_cfl(DIMS, ksp_dims, kspace_data);

	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, grid_dims);

		pattern = anon_cfl("", DIMS, pat_dims);

		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_data);

		md_copy(DIMS, grid_dims, k_grid_data, kspace_data, CFL_SIZE);

		unmap_cfl(DIMS, ksp_dims, kspace_data);
	}

	if (conf.k_filter) {

		long map_dims[DIMS];
		md_select_dims(DIMS, FFT_FLAGS, map_dims, pat_dims);

		long map_strs[DIMS];
		md_calc_strides(DIMS, map_strs, map_dims, CFL_SIZE);

		long pat_strs[DIMS];
		md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

		complex float* filter = md_alloc(DIMS, map_dims, CFL_SIZE);

		switch (conf.k_filter_type) {

		case EF1:
			edge_filter1(map_dims, filter, kfilter_strength);
			break;

		case EF2:
			edge_filter2(map_dims, filter, kfilter_strength);
			break;
		}

		md_zadd2(DIMS, pat_dims, pat_strs, pattern, pat_strs, pattern, map_strs, filter);

		md_free(filter);
	}

	// read initialization file

	long init_dims[DIMS] = { [0 ... DIMS-1] = 1 };
	complex float* init = (NULL != init_file) ? load_cfl(init_file, DIMS, init_dims) : NULL;

	assert(md_check_bounds(DIMS, 0, img_dims, init_dims));

        // Load passed B1

        const complex float* b1 = NULL;
	long b1_dims[DIMS];

	if (NULL != input_b1) {

		b1 = load_cfl(input_b1, DIMS, b1_dims);

		assert(md_check_bounds(DIMS, FFT_FLAGS, grid_dims, b1_dims));
	}

	// scaling

	if (normalize_scaling) {

		scaling /= md_znorm(DIMS, grid_dims, k_grid_data);
		scaling_psf /= md_znorm(DIMS, pat_dims, pattern);
	}

	if (1. != scaling) {

		debug_printf(DP_INFO, "Scaling: %f\n", scaling);
		md_zsmul(DIMS, grid_dims, k_grid_data, k_grid_data, scaling);
	}

	if (1. != scaling_psf) {

		debug_printf(DP_INFO, "Scaling_psf: %f\n", scaling_psf);
		md_zsmul(DIMS, pat_dims, pattern, pattern, scaling_psf);
	}


	// mask

        // Idea:        Speed up md-function based nlops by skipping zero parts,
	//              not required for Bloch model, because other_conf.fov_reduction_factor
	//              constrains the k-space coverage there

	if (-1. == restrict_fov) {

		mask = md_alloc(DIMS, msk_dims, CFL_SIZE);

		md_zfill(DIMS, msk_dims, mask, 1.);

		data.other.fov_reduction_factor = 1;

	} else {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;

		mask = compute_mask(DIMS, msk_dims, restrict_dims);

		data.other.fov_reduction_factor = restrict_fov;

                if (MDB_BLOCH != conf.mode)
		        md_zmul2(DIMS, img_dims, img_strs, img, img_strs, img, msk_strs, mask);
	}

        // Scale parameter maps

        long tmp_dims[DIMS];
        md_select_dims(DIMS, FFT_FLAGS|MAPS_FLAG|TIME_FLAG|SLICE_FLAG|TIME2_FLAG, tmp_dims, grid_dims);

        complex float* tmp = md_alloc(DIMS, tmp_dims, CFL_SIZE);

        long pos[DIMS] = { [0 ... DIMS - 1] = 0 };

	assert(img_dims[COEFF_DIM] <= (long)ARRAY_SIZE(data.other.scale));

	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {

		pos[COEFF_DIM] = i;

		md_copy_block(DIMS, pos, tmp_dims, tmp, img_dims, img, CFL_SIZE);

		md_zsmul(DIMS, tmp_dims, tmp, tmp, data.other.initval[i] / (data.other.scale[i] ?: 1));

		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp, CFL_SIZE);
	}

        // Transform B1 map from image to k-space and add k-space to initialization array (img)

	if (MDB_BLOCH == conf.mode) {

		pos[COEFF_DIM] = 3;

		const struct linop_s* linop_fftc = linop_fftc_create(DIMS, tmp_dims, FFT_FLAGS);

		md_copy_block(DIMS, pos, tmp_dims, tmp, img_dims, img, CFL_SIZE);

		linop_forward_unchecked(linop_fftc, tmp, tmp);

		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp, CFL_SIZE);

		linop_free(linop_fftc);
	}

#ifdef  USE_CUDA
	if (conf.use_gpu) {

//		cuda_use_global_memory();

		complex float* kspace_gpu = md_alloc_gpu(DIMS, grid_dims, CFL_SIZE);

		md_copy(DIMS, grid_dims, kspace_gpu, k_grid_data, CFL_SIZE);

		complex float* TI_gpu = md_alloc_gpu(DIMS, TI_dims, CFL_SIZE);

		md_copy(DIMS, TI_dims, TI_gpu, TI, CFL_SIZE);

		moba_recon(&conf, &data, dims, img, sens, pattern, mask, TI_gpu, b1, kspace_gpu, init);

		md_free(kspace_gpu);
		md_free(TI_gpu);

	} else
#endif
	moba_recon(&conf, &data, dims, img, sens, pattern, mask, TI, b1, k_grid_data, init);

        // Rescale estimated parameter maps

	for (int i = 0; i < img_dims[COEFF_DIM]; i++) {

		pos[COEFF_DIM] = i;

		md_copy_block(DIMS, pos, tmp_dims, tmp, img_dims, img, CFL_SIZE);

		md_zsmul(DIMS, tmp_dims, tmp, tmp, (data.other.scale[i] ?: 1.));

		md_copy_block(DIMS, pos, img_dims, img, tmp_dims, tmp, CFL_SIZE);
	}

        md_free(tmp);
	md_free(mask);

	unmap_cfl(DIMS, coil_dims, sens);
	unmap_cfl(DIMS, pat_dims, pattern);
	unmap_cfl(DIMS, grid_dims, k_grid_data);
	unmap_cfl(DIMS, img_dims, img);
	unmap_cfl(DIMS, single_map_dims, single_map);
	unmap_cfl(DIMS, TI_dims, TI);

	if (NULL != traj_file)
		unmap_cfl(DIMS, traj_dims, traj);

	if (NULL != init_file)
		unmap_cfl(DIMS, init_dims, init);

        if(NULL != input_b1)
		unmap_cfl(DIMS, b1_dims, b1);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);

	return 0;
}

