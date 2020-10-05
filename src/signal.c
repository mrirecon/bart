/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


#include <math.h>
#include <complex.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "simu/signals.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char usage_str[] = "<basis-functions>";
static const char help_str[] = "Analytical simulation tool.";





int main_signal(int argc, char* argv[])
{
	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = 100;

	enum seq_type { BSSFP, FLASH, TSE, MOLLI };
	enum seq_type seq = FLASH;

	bool IR = false;
	bool IR_SS = false;
	float FA = -1.;
	float TR = -1.;
	float TE = -1.;
	float time_T1relax = -1.; // second
	long Hbeats = -1;

	float T1[3] = { 500., 1500., 10 };
	float T2[3] = {  50.,  150., 10 };

	const struct opt_s opts[] = {

		OPT_SELECT('F', enum seq_type, &seq, FLASH, "FLASH"),
		OPT_SELECT('B', enum seq_type, &seq, BSSFP, "bSSFP"),
		OPT_SELECT('T', enum seq_type, &seq, TSE, "TSE"),
		OPT_SELECT('M', enum seq_type, &seq, MOLLI, "MOLLI"),
		OPT_SET('I', &IR, "inversion recovery"),
		OPT_SET('s', &IR_SS, "inversion recovery starting from steady state"),
		OPT_FLVEC3('1', &T1, "min:max:N", "range of T1s"),
		OPT_FLVEC3('2', &T2, "min:max:N", "range of T2s"),
		OPT_FLOAT('r', &TR, "TR", "repetition time"),
		OPT_FLOAT('e', &TE, "TE", "echo time"),
		OPT_FLOAT('f', &FA, "FA", "flip ange"),
		OPT_FLOAT('t', &time_T1relax, "T1 relax", "T1 relax period (second) for MOLLI"),
		OPT_LONG('n', &dims[TE_DIM], "n", "number of measurements"),
		OPT_LONG('b', &Hbeats, "heart beats", "number of heart beats for MOLLI"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	struct signal_model parm;

	if ((!IR) && (BSSFP == seq))
		error("only IR signal supported for bSSFP");

	switch (seq) {

	case FLASH: parm = signal_looklocker_defaults; break;
	case BSSFP: parm = signal_IR_bSSFP_defaults; break;
	case TSE:   parm = signal_TSE_defaults; break;
	case MOLLI: parm = signal_looklocker_defaults; break;

	default: error("sequence type not supported");
	}

	if (-1. != FA)
		parm.fa = FA * M_PI / 180.;

	if (-1. != TR)
		parm.tr = TR;

        parm.ir = IR;
	parm.ir_ss = IR_SS;

	assert(!(parm.ir && parm.ir_ss));

	// if (-1 != TE)
	// 	parm.te = TE;

	dims[COEFF_DIM] = truncf(T1[2]);
	dims[COEFF2_DIM] = truncf(T2[2]);

	if ((dims[TE_DIM] < 1) || (dims[COEFF_DIM] < 1) || (dims[COEFF2_DIM] < 1))
		error("invalid parameter range");

	complex float* signals = create_cfl(argv[1], DIMS, dims);

	long dims1[DIMS];
	md_select_dims(DIMS, TE_FLAG, dims1, dims);

	long pos[DIMS] = { 0 };
	int N = dims[TE_DIM];

	do {
		parm.t1 = T1[0] + (T1[1] - T1[0]) / T1[2] * (float)pos[COEFF_DIM];
		parm.t2 = T2[0] + (T2[1] - T2[0]) / T2[2] * (float)pos[COEFF2_DIM];

		complex float out[N];

		switch (seq) {

		case FLASH: looklocker_model(&parm, N, out); break;
		case BSSFP: IR_bSSFP_model(&parm, N, out); break;
		case TSE:   TSE_model(&parm, N, out); break;
		case MOLLI: MOLLI_model(&parm, N, Hbeats, time_T1relax, out); break;

		default: assert(0);
		}

		md_copy_block(DIMS, pos, dims, signals, dims1, out, CFL_SIZE);

	} while(md_next(DIMS, dims, ~TE_FLAG, pos));

	unmap_cfl(DIMS, dims, signals);

	return 0;
}


