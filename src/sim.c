/* Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author: Nick Scholand, Martin Juschitz
 */

#include <math.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "seq/pulse.h"

#include "simu/bloch.h"
#include "simu/simulation.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static const char help_str[] = "simulation tool";


// FIXME: Turn off sensitivity analysis if derivatives are not asked for
static void perform_bloch_simulation(int N, struct sim_data* data, long mdims[N], complex float* mxy, long ddims[N], complex float* deriv)     // 4 Derivatives: dR1, dM0, dR2, dB1
{
	int D = ddims[READ_DIM];
	int T = ddims[TE_DIM];

	int A = 3;

	float m[T][data->voxel.P][3];
	float sa_r1[T][data->voxel.P][3];
	float sa_r2[T][data->voxel.P][3];
	float sa_m0[T][data->voxel.P][3];
	float sa_b1[T][1][3];
	float sa_Om[T][data->voxel.P][3];
	float sa_k[T][data->voxel.P][3];	// [T][data->voxel.P - 1][A] is empty for k and Om

	bloch_simulation2(data, T, data->voxel.P, &m, &sa_r1, &sa_r2, &sa_m0, &sa_b1, &sa_k, &sa_Om);

	long pos[DIMS];
	md_copy_dims(DIMS, pos, ddims);

	long dstrs[DIMS];
	md_calc_strides(N, dstrs, ddims, CFL_SIZE);

	long mstrs[DIMS];
	md_calc_strides(N, mstrs, mdims, CFL_SIZE);

	long ind = 0;

	for (int d = 0; d < D; d++) {

		pos[READ_DIM] = d;

		for (int i = 0; i < T; i++) {

			// Calculate spatial position and save data

			pos[TE_DIM] = i;

			for (int p = 0; p < data->voxel.P; p++) {

				pos[MAPS_DIM] = 0;
				pos[ITER_DIM] = p;
				ind = md_calc_offset(N, mstrs, pos) / (long)CFL_SIZE;

				// M = M_x + i M_y
				mxy[ind] = (A == D) ? m[i][p][d] : (m[i][p][0] + 1.i * m[i][p][1]);

				if (NULL == deriv)
					continue;

				ind = md_calc_offset(N, dstrs, pos) / (long)CFL_SIZE;
				deriv[ind] = (A == D) ? sa_r1[i][p][d] : (sa_r1[i][p][0] + 1.i * sa_r1[i][p][1]);

				pos[MAPS_DIM] = 1;
				ind = md_calc_offset(N, dstrs, pos) / (long)CFL_SIZE;
				deriv[ind] = (A == D) ? sa_m0[i][p][d] : (sa_m0[i][p][0] + 1.i * sa_m0[i][p][1]);

				pos[MAPS_DIM] = 2;
				ind = md_calc_offset(N, dstrs, pos) / (long)CFL_SIZE;
				deriv[ind] = (A == D) ? sa_r2[i][p][d] : (sa_r2[i][p][0] + 1.i * sa_r2[i][p][1]);

				pos[MAPS_DIM] = 3;
				ind = md_calc_offset(N, dstrs, pos) / (long)CFL_SIZE;

				if (0 == p)
					deriv[ind] = (A == D) ? sa_b1[i][0][d] : (sa_b1[i][0][0] + 1.i * sa_b1[i][0][1]);

				if (p < data->voxel.P - 1) {

					pos[MAPS_DIM] = 4;
					ind = md_calc_offset(N, dstrs, pos) / (long)CFL_SIZE;
					deriv[ind] = (A == D) ? sa_k[i][p][d] : (sa_k[i][p][0] + 1.i * sa_k[i][p][1]);

					pos[MAPS_DIM] = 5;
					ind = md_calc_offset(N, dstrs, pos) / (long)CFL_SIZE;
					deriv[ind] = (A == D) ? sa_Om[i][p][d] : (sa_Om[i][p][0] + 1.i * sa_Om[i][p][1]);
				}
			}
		}
	}
}


int main_sim(int argc, char* argv[argc])
{
	const char* out_signal = NULL;
	const char* out_deriv = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_signal, "signal: Mxy"),
		ARG_OUTFILE(false, &out_deriv, "Partial derivatives: dR1, dM0, dR2, dB1"),
	};

	struct sim_data data;
	data.seq = simdata_seq_defaults;
	data.voxel = simdata_voxel_defaults;
	data.pulse = simdata_pulse_defaults;
	data.pulse.sinc = pulse_sinc_defaults;
	data.pulse.hs = pulse_hypsec_defaults;
	data.grad = simdata_grad_defaults;
	data.other = simdata_other_defaults;
	data.cest = simdata_cest_defaults;

	float T1[3] = { WATER_T1, WATER_T1, 1 };
	float T2[3] = { WATER_T2, WATER_T2, 1 };

	// Further pool parameters
	float T1pools[4] = { 0., 0., 0., 0. };
	float T2pools[4] = { 0., 0., 0., 0. };
	float Ompools[4] = { 0., 0., 0., 0. };
	float M0pools[4] = { 0., 0., 0., 0. };
	float kpools[4] = { 0., 0., 0., 0. };

	bool split_dim = false;

	struct opt_s seq_opts[] = {

		/* Sequence Specific Parameters */
		OPTL_SELECT(0, "BSSFP", enum sim_seq, &(data.seq.seq_type), SEQ_BSSFP, "bSSFP"),
		OPTL_SELECT(0, "IR-BSSFP", enum sim_seq, &(data.seq.seq_type), SEQ_IRBSSFP, "Inversion-Recovery bSSFP"),
		OPTL_SELECT(0, "FLASH", enum sim_seq, &(data.seq.seq_type), SEQ_FLASH, "FLASH"),
		OPTL_SELECT(0, "IR-FLASH", enum sim_seq, &(data.seq.seq_type), SEQ_IRFLASH, "Inversion-Recovery FLASH"),
		OPTL_SELECT(0, "CEST", enum sim_seq, &(data.seq.seq_type), SEQ_CEST, "CEST"),
		OPTL_FLOAT(0, "TR", &(data.seq.tr), "float", "Repetition time [s]"),
		OPTL_FLOAT(0, "TE", &(data.seq.te), "float", "Echo time [s]"),
		OPTL_PINT(0, "Nspins", &(data.seq.spin_num), "int", "Number of averaged spins"),
		OPTL_PINT(0, "Nrep", &(data.seq.rep_num), "int", "Number of repetitions"),
		OPTL_SET(0, "pinv", &(data.seq.perfect_inversion), "Use perfect inversions"),
		OPTL_FLOAT(0, "ipl", &(data.seq.inversion_pulse_length), "float", "Inversion Pulse Length [s]"),
		OPTL_FLOAT(0, "isp", &(data.seq.inversion_spoiler), "float", "Inversion Spoiler Gradient Length [s]"),
		OPTL_FLOAT(0, "ppl", &(data.seq.prep_pulse_length), "float", "Preparation Pulse Length [s]"),
		OPTL_PINT(0, "av-spokes", &(data.seq.averaged_spokes), "", "Number of averaged consecutive spokes"),

		/* Pulse Specific Parameters */
		OPTL_FLOAT(0, "Trf", &(data.pulse.rf_end), "float", "Pulse Duration [s]"), /* Assumes to start at t=0 */
		OPTL_FLOAT(0, "FA", &(data.pulse.sinc.super.flipangle), "float", "Flipangle [deg]"),
		OPTL_FLOAT(0, "BWTP", &(data.pulse.sinc.bwtp), "float", "Bandwidth-Time-Product"),

		/* Voxel Specific Parameters */
		OPTL_FLOAT(0, "off", &(data.voxel.w), "float", "Off-Resonance [rad/s]"),

		/* Slice Profile Parameters */
		OPTL_FLOAT(0, "sl-grad", &(data.grad.sl_gradient_strength), "float", "Strength of Slice Selection Gradient [T/m]"),
		OPTL_FLOAT(0, "slice-thickness", &(data.seq.slice_thickness), "float", "Thickness of simulated slice [m]."),

	};
	const int N_seq_opts = ARRAY_SIZE(seq_opts);

	struct opt_s other_opts[] = {

		OPTL_FLOAT(0, "ode-tol", &(data.other.ode_tol), "", "ODE tolerance value [def: 1e-5]"),
		OPTL_FLOAT(0, "stm-tol", &(data.other.stm_tol), "", "STM tolerance value [def: 1e-6]"),
		OPTL_FLOAT(0, "sampling-rate", &(data.other.sampling_rate), "", "Sampling rate of RF pulse used for ROT simulation in Hz [def: 1e6 Hz]"),
	};
	const int N_other_opts = ARRAY_SIZE(other_opts);


	struct opt_s pool_opts[] = {

		OPT_PINT('P', &(data.voxel.P), "int", "Number of pools"),
		OPTL_FLVEC4(0, "T1", &T1pools, "<2nd pool>:<3rd pool>:<4th pool>:<5th pool>", "T1 values for further pools"),
		OPTL_FLVEC4(0, "T2", &T2pools, "<2nd pool>:<3rd pool>:<4th pool>:<5th pool>", "T2 values for further pools"),
		OPTL_FLVEC4(0, "Om", &Ompools, "<2nd pool>:<3rd pool>:<4th pool>:<5th pool>", "Om values for further pools"),
		OPTL_FLVEC4(0, "M0", &M0pools, "<2nd pool>:<3rd pool>:<4th pool>:<5th pool>", "M0 values for further pools"),
		OPTL_FLVEC4(0, "k",  &kpools, "<k1>:<k2>:<k3>:<k4>", "k values for further pools"),
	};
	const int N_pool_opts = ARRAY_SIZE(pool_opts);

	struct opt_s cest_opts[] = {

		OPTL_FLOAT(0, "b1", &(data.cest.b1_amp), "float", "B1 amplitude [mu T (RMS)]"),
		OPTL_FLOAT(0, "b0", &(data.cest.b0), "float", "B0 [T]"),
		OPTL_FLOAT(0, "gamma", &(data.cest.gamma), "float", "Gyromagnetic ratio [Mhz/T]"),
		OPTL_FLOAT(0, "max", &(data.cest.off_start), "float", "Max offset [ppm]"),
		OPTL_FLOAT(0, "min", &(data.cest.off_stop), "float", "Min offset [ppm]"),
		OPTL_PINT(0, "n_p", &(data.cest.n_pulses), "int", "Number of pulses"),
		OPTL_FLOAT(0, "t_d", &(data.cest.t_d), "float", "Interpulse delay [s]"),
		OPTL_FLOAT(0, "t_pp", &(data.cest.t_pp), "float", "Post-preparation delay [s]"),
		OPTL_SET(0, "ref_scan", &(data.cest.ref_scan), "Use reference scan"),
		OPTL_FLOAT(0, "ref_scan_ppm", &(data.cest.ref_scan_ppm), "float", "Offset for ref. scan [ppm]"),
	};
	const int N_cest_opts = ARRAY_SIZE(cest_opts);

	const struct opt_s opts[] = {

		OPTL_FLVEC3('1', "T1", &T1, "min:max:N", "range of T1 values"),
		OPTL_FLVEC3('2', "T2", &T2, "min:max:N", "range of T2 values"),
		OPTL_SELECT(0, "BLOCH", enum sim_model, &(data.seq.model), MODEL_BLOCH, "Bloch Equations (default)"),
		OPTL_SELECT(0, "BMC", enum sim_model, &(data.seq.model), MODEL_BMC, "Bloch-McConnell Equations"),
		OPTL_SELECT(0, "ROT", enum sim_type, &(data.seq.type), SIM_ROT, "discretized simulation based on rotational matrices"),
		OPTL_SELECT(0, "ODE", enum sim_type, &(data.seq.type), SIM_ODE, "ordinary differential equation solver (default)"),
		OPTL_SELECT(0, "STM", enum sim_type, &(data.seq.type), SIM_STM, "solver based on state-transition matrices"),
		OPTL_SET(0, "split-dim", &split_dim, "split magnetization into x, y, and z component"),
		OPTL_SUBOPT(0, "seq", "...", "configure sequence parameter", N_seq_opts, seq_opts),
		OPTL_SUBOPT(0, "other", "...", "configure other parameters", N_other_opts, other_opts),
		OPTL_SUBOPT(0, "pool", "...", "configure parameters for 2nd->5th pool", N_pool_opts, pool_opts),
		OPTL_SUBOPT(0, "CEST", "...", "configure parameters for CEST", N_cest_opts, cest_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	// Tests for validity of input parameter

	if (MODEL_BMC == data.seq.model)
		assert(1 < data.voxel.P);

	// Define output dimensions for signal

	long mdims[DIMS] = { [0 ... DIMS - 1] = 1 };

	if (split_dim)
		mdims[READ_DIM] = 3; // (x, y, z)

	mdims[TE_DIM] = data.seq.rep_num;

	mdims[COEFF_DIM] = truncf(T1[2]);
	mdims[COEFF2_DIM] = truncf(T2[2]);
	mdims[ITER_DIM] = data.voxel.P;

	if ((mdims[TE_DIM] < 1) || (mdims[COEFF_DIM] < 1) || (mdims[COEFF2_DIM] < 1))
		error("invalid parameter range\n");

	// Allocate output file for signal and optional derivatives

	complex float* signals = create_cfl(out_signal, DIMS, mdims);

	long ddims[DIMS] = { [0 ... DIMS - 1] = 1 };

	md_copy_dims(DIMS, ddims, mdims);
	ddims[MAPS_DIM] = (1 == data.voxel.P) ? 4 : 6; // [dR1, dM0, dR2, dB1], dOm, dk

	complex float* deriv = NULL;

	if (NULL != out_deriv)
		deriv = create_cfl(out_deriv, DIMS, ddims);

	// Temporary dimensions for derivative and magnetization

	long tmdims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG|COEFF2_FLAG), tmdims, mdims);

	long tddims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG|COEFF2_FLAG), tddims, ddims);


	// Allocate temporary magnetization and derivative arrays

	complex float* tm = md_calloc(DIMS, tmdims, CFL_SIZE);
	complex float* td = md_calloc(DIMS, tddims, CFL_SIZE);


	// Run all simulations and store signal and optional derivatives

	long pos[DIMS] = { };

	// Starting time of simulation
	double start = timestamp();
	do {
		data.voxel.r1[0] = 1. / (T1[0] + (T1[1] - T1[0]) / T1[2] * (float)pos[COEFF_DIM]);
		data.voxel.r2[0] = 1. / (T2[0] + (T2[1] - T2[0]) / T2[2] * (float)pos[COEFF2_DIM]);

		for (int i = 1; i < data.voxel.P; i++) {

			if (0 != T1pools[i - 1])
				data.voxel.r1[i] = 1 / T1pools[i - 1];

			if (0 != T2pools[i - 1])
				data.voxel.r2[i] = 1 / T2pools[i - 1];

			data.voxel.m0[i] = M0pools[i - 1];
			data.voxel.Om[i] = Ompools[i - 1];
			data.voxel.k[i - 1] = kpools[i - 1];
		}

		perform_bloch_simulation(DIMS, &data, tmdims, tm, tddims, td);

		md_copy_block(DIMS, pos, mdims, signals, tmdims, tm, CFL_SIZE);

		if (NULL != deriv)
			md_copy_block(DIMS, pos, ddims, deriv, tddims, td, CFL_SIZE);

	} while (md_next(DIMS, mdims, ~(READ_FLAG|MAPS_FLAG|TE_FLAG|ITER_FLAG), pos));

	// End time of simulation
	double end = timestamp();

	debug_printf(DP_INFO, "%f\n", end - start);

	md_free(tm);
	md_free(td);

	unmap_cfl(DIMS, mdims, signals);
	unmap_cfl(DIMS, ddims, deriv);

	return 0;
}

