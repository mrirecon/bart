/* Copyright 2022. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
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

#include "simu/bloch.h"
#include "simu/pulse.h"
#include "simu/simulation.h"
#include "simu/slice_profile.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static const char help_str[] = "simulation tool";


// FIXME: Turn of sensitivity analysis if derivatives are not asked for
static void perform_bloch_simulation(int N, struct sim_data* data, const complex float* slice, long mdims[N], complex float* mxy, long ddims[N], complex float* deriv)     // 4 Derivatives: dR1, dM0, dR2, dB1
{
        (void) mdims;

        int T = ddims[TE_DIM];

        float m[T][3];
        float sa_r1[T][3];
        float sa_r2[T][3];
        float sa_m0[T][3];
        float sa_b1[T][3];

        bloch_simulation(data, slice, T, &m, &sa_r1, &sa_r2, &sa_m0, &sa_b1);

        long pos[DIMS];
        md_copy_dims(DIMS, pos, ddims);

        long dstrs[DIMS];
        md_calc_strides(N, dstrs, ddims, CFL_SIZE);

        long ind = 0;

        for (int i = 0; i < T; i++) {

                // Calculate spatial position and save data

                pos[TE_DIM] = i;
                pos[MAPS_DIM] = 0;

                ind = md_calc_offset(N, dstrs, pos) / CFL_SIZE;

                mxy[i] = m[i][1] + 1.i * m[i][0];

                if (NULL != deriv) {

                        // dR1
                        deriv[ind] = sa_r1[i][1] + 1.i * sa_r1[i][0];

                        // dM0
                        pos[MAPS_DIM] = 1;
                        ind = md_calc_offset(N, dstrs, pos) / CFL_SIZE;
                        deriv[ind] = sa_m0[i][1] + 1.i * sa_m0[i][0];

                        // dR2
                        pos[MAPS_DIM] = 2;
                        ind = md_calc_offset(N, dstrs, pos) / CFL_SIZE;
                        deriv[ind] = sa_r2[i][1] + 1.i * sa_r2[i][0];

                        // dB1
                        pos[MAPS_DIM] = 3;
                        ind = md_calc_offset(N, dstrs, pos) / CFL_SIZE;
                        deriv[ind] = sa_b1[i][1] + 1.i * sa_b1[i][0];
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
        data.pulse.hs = hs_pulse_defaults;
	data.grad = simdata_grad_defaults;
	data.tmp = simdata_tmp_defaults;
        data.other = simdata_other_defaults;

        float T1[3] = { WATER_T1, WATER_T1, 1 };
	float T2[3] = { WATER_T2, WATER_T2, 1 };

        struct opt_s seq_opts[] = {

                /* Sequence Specific Parameters */
                OPTL_SELECT(0, "bssfp", enum sim_seq, &(data.seq.seq_type), SEQ_BSSFP, "bSSFP"),
                OPTL_SELECT(0, "ir-bssfp", enum sim_seq, &(data.seq.seq_type), SEQ_IRBSSFP, "Inversion-Recovery bSSFP"),
                OPTL_SELECT(0, "flash", enum sim_seq, &(data.seq.seq_type), SEQ_FLASH, "FLASH"),
                OPTL_SELECT(0, "ir-flash", enum sim_seq, &(data.seq.seq_type), SEQ_IRFLASH, "Inversion-Recovery FLASH"),
                OPTL_FLOAT(0, "tr", &(data.seq.tr), "float", "Repetition time [s]"),
                OPTL_FLOAT(0, "te", &(data.seq.te), "float", "Echo time [s]"),
                OPTL_INT(0, "nspins", &(data.seq.spin_num), "int", "Number of averaged spins"),
                OPTL_INT(0, "nrep", &(data.seq.rep_num), "int", "Number of repetitions"),
                OPTL_SET(0, "pinv", &(data.seq.perfect_inversion), "Use perfect inversions"),
                OPTL_FLOAT(0, "ipl", &(data.seq.inversion_pulse_length), "float", "Inversion Pulse Length [s]"),
                OPTL_FLOAT(0, "isp", &(data.seq.inversion_spoiler), "float", "Inversion Spoiler Gradient Length [s]"),
                OPTL_FLOAT(0, "ppl", &(data.seq.prep_pulse_length), "float", "Preparation Pulse Length [s]"),
                OPTL_INT(0, "av-spokes", &(data.seq.averaged_spokes), "", "Number of averaged consecutive spokes"),
                OPTL_INT(0, "slice-profile-spins", &(data.seq.slice_profile_spins), "", "Number of spins in the slice-profile"),

                /* Pulse Specific Parameters */
                OPTL_FLOAT(0, "trf", &(data.pulse.rf_end), "float", "Pulse Duration [s]"), /* Assumes to start at t=0 */
                OPTL_FLOAT(0, "fa", &(data.pulse.flipangle), "float", "Flipangle [deg]"),
                OPTL_FLOAT(0, "bwtp", &(data.pulse.bwtp), "float", "Bandwidth-Time-Product"),

                /* Voxel Specific Parameters */
                OPTL_FLOAT(0, "off", &(data.voxel.w), "float", "Off-Resonance [rad/s]"),

                /* Gradient Specific Parameters */
                OPTL_FLOAT(0, "mom-sl", &(data.grad.mom_sl), "float", "Slice Selection Gradient Moment [rad/s]"),

        };
        const int N_seq_opts = ARRAY_SIZE(seq_opts);

        struct opt_s other_opts[] = {

                OPTL_FLOAT(0, "ode-tol", &(data.other.ode_tol), "", "ODE tolerance value [def: 10E-6]"),
                OPTL_FLOAT(0, "sampling-rate", &(data.other.sampling_rate), "", "Sampling rate of RF pulse used for ROT simulation in Hz [def: 10E5 Hz]"),
        };
        const int N_other_opts = ARRAY_SIZE(other_opts);


	const struct opt_s opts[] = {

                OPTL_FLVEC3('1',        "T1",	&T1, 			"min:max:N", "range of T1 values"),
		OPTL_FLVEC3('2',	"T2",   &T2, 			"min:max:N", "range of T2 values"),
                OPTL_SELECT(0, "ROT", enum sim_type, &(data.seq.type), SIM_ROT, "homogeneously discretized simulation based on rotational matrices"),
                OPTL_SELECT(0, "ODE", enum sim_type, &(data.seq.type), SIM_ODE, "full ordinary differential equation solver based simulation (default)"),
                OPTL_SELECT(0, "STM", enum sim_type, &(data.seq.type), SIM_STM, "state-transition matrix based simulation"),
                OPTL_SUBOPT(0, "seq", "...", "configure sequence parameter", N_seq_opts, seq_opts),
                OPTL_SUBOPT(0, "other", "...", "configure other parameters", N_other_opts, other_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);


        // Define output dimensions for signal

        long mdims[DIMS] = { [0 ... DIMS - 1] = 1 };

	mdims[TE_DIM] = data.seq.rep_num;

	mdims[COEFF_DIM] = truncf(T1[2]);
	mdims[COEFF2_DIM] = truncf(T2[2]);

	if ((mdims[TE_DIM] < 1) || (mdims[COEFF_DIM] < 1) || (mdims[COEFF2_DIM] < 1))
		error("invalid parameter range");


        // Approximate slice profile

        long spdims[DIMS] = { [0 ... DIMS - 1] = 1 };
	complex float* slice = NULL;

	if (1 != data.seq.slice_profile_spins) {

		assert((1 == data.seq.spin_num) || (data.seq.spin_num == data.seq.slice_profile_spins));

		data.seq.spin_num = data.seq.slice_profile_spins;

		spdims[READ_DIM] = data.seq.spin_num;	// FIXME: Why read?
		slice = md_alloc(DIMS, spdims, CFL_SIZE);


		sinc_pulse_init(&data.pulse, data.pulse.rf_start, data.pulse.rf_end, data.pulse.flipangle, data.pulse.phase, data.pulse.bwtp, data.pulse.alpha); // FIXME
		slice_profile_fourier(DIMS, spdims, slice, &data.pulse);
        }

        // Allocate output file for signal and optional derivatives

	complex float* signals = create_cfl(out_signal, DIMS, mdims);

        long ddims[DIMS] = { [0 ... DIMS - 1] = 1 };

        md_copy_dims(DIMS, ddims, mdims);
        ddims[MAPS_DIM] = 4;    // dR1, dM0, dR2, dB1

        complex float* deriv = NULL;

        if (NULL != out_deriv)
                deriv = create_cfl( out_deriv, DIMS, ddims);

        // Temporary dimensions for derivative and magnetization

	long tmdims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG|COEFF2_FLAG), tmdims, mdims);

	long tddims[DIMS];
	md_select_dims(DIMS, ~(COEFF_FLAG|COEFF2_FLAG), tddims, ddims);


        // Allocate temporary magnetization and derivative arrays

	complex float* tm = md_calloc(DIMS, tmdims, CFL_SIZE);
	complex float* td = md_calloc(DIMS, tddims, CFL_SIZE);


        // Run all simulations and store signal and optional derivatives

        long pos[DIMS] = { 0 };

	do {
		data.voxel.r1 = 1. / (T1[0] + (T1[1] - T1[0]) / T1[2] * (float)pos[COEFF_DIM]);
        	data.voxel.r2 = 1. / (T2[0] + (T2[1] - T2[0]) / T2[2] * (float)pos[COEFF2_DIM]);

                perform_bloch_simulation(DIMS, &data, slice, tmdims, tm, tddims, td);

		md_copy_block(DIMS, pos, mdims, signals, tmdims, tm, CFL_SIZE);

                if (NULL != deriv)
                        md_copy_block(DIMS, pos, ddims, deriv, tddims, td, CFL_SIZE);

	} while(md_next(DIMS, mdims, ~TE_FLAG, pos));

        md_free(tm);
        md_free(td);

	unmap_cfl(DIMS, mdims, signals);

        if (NULL != out_deriv)
                unmap_cfl(DIMS, ddims, deriv);

	if (NULL != slice)
		md_free(slice);

	return 0;
}


