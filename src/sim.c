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

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "simu/bloch.h"
#include "simu/pulse.h"
#include "simu/simulation.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static void perform_bloch_simulation(struct sim_data* data, int N, complex float out[N])
{
        float m[N][3];
        float sa_r1[N][3];
        float sa_r2[N][3];
        float sa_m0[N][3];
        float sa_b1[N][3];

        bloch_simulation2(data, m, sa_r1, sa_r2, sa_m0, sa_b1);

        for (int i = 0; i < N; i++)
                out[i] = m[i][1] + m[i][0] * I;
}


static const char help_str[] = "simulation tool";


int main_sim(int argc, char* argv[argc])
{
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "basis-functions"),
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
        };
        const int N_other_opts = ARRAY_SIZE(other_opts);


	const struct opt_s opts[] = {

                OPTL_FLVEC3('1',        "T1",	&T1, 			"min:max:N", "range of T1 values"),
		OPTL_FLVEC3('2',	"T2",   &T2, 			"min:max:N", "range of T2 values"),
                OPTL_SELECT(0, "ODE", enum sim_type, &(data.seq.type), SIM_ODE, "full ordinary differential equation solver based simulation (default)"),
                OPTL_SELECT(0, "STM", enum sim_type, &(data.seq.type), SIM_STM, "state-transition matrix based simulation"),
                OPTL_SUBOPT(0, "seq", "...", "configure sequence parameter", N_seq_opts, seq_opts),
                OPTL_SUBOPT(0, "other", "...", "configure other parameters", N_other_opts, other_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

        // Define output dimensions
        long dims[DIMS] = { [0 ... DIMS - 1] = 1 };

	dims[TE_DIM] = data.seq.rep_num;
	dims[COEFF_DIM] = truncf(T1[2]);
	dims[COEFF2_DIM] = truncf(T2[2]);

	if ((dims[TE_DIM] < 1) || (dims[COEFF_DIM] < 1) || (dims[COEFF2_DIM] < 1))
		error("invalid parameter range");


        // Allocate output file
	complex float* signals = create_cfl(out_file, DIMS, dims);

	long dims1[DIMS];
	md_select_dims(DIMS, TE_FLAG, dims1, dims);

	long pos[DIMS] = { 0 };
	int N = dims[TE_DIM];


        // Run all simulations and store magnetization
	do {
		data.voxel.r1 = 1. / (T1[0] + (T1[1] - T1[0]) / T1[2] * (float)pos[COEFF_DIM]);
        	data.voxel.r2 = 1. / (T2[0] + (T2[1] - T2[0]) / T2[2] * (float)pos[COEFF2_DIM]);

		complex float out[N];

                perform_bloch_simulation(&data, N, out);

		md_copy_block(DIMS, pos, dims, signals, dims1, out, CFL_SIZE);

	} while(md_next(DIMS, dims, ~TE_FLAG, pos));

	unmap_cfl(DIMS, dims, signals);

	return 0;
}


