/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Volkert Roeloffs
 */

#include <math.h>
#include <complex.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "simu/epg.h"
#include "simu/crb.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char usage_str[] = "<signal intensity> [<configuration states> [<(rel.) signal derivatives> [<configuration derivatives>]]]";
static const char help_str[] = "Simulate MR pulse sequence based on Extended Phase Graphs (EPG)";

int main_epg(int argc, char* argv[argc])
{

	enum seq_type {CPMG, FMSSFP, HYPER, FLASH, SPINECHO, BSSFP};
	enum seq_type seq = CPMG;

	bool save_states = false;
	bool save_sigder = false;
	bool save_statesder = false;

	float T1 =  0.800;
	float T2 =  0.050;
	float TR =  0.005;
	float TE =  0.010;
	float FA =  15.0;
	float offres = 0.0;
	float B1 =   1.0;
	long  SP =     0;
	long   N =    10;
	long unknowns = 3;
	long verbose = 0;
	const struct opt_s opts[] = {

		OPT_SELECT('C', enum seq_type, &seq, CPMG, "CPMG"),
		OPT_SELECT('M', enum seq_type, &seq, FMSSFP, "fmSSFP"),
		OPT_SELECT('H', enum seq_type, &seq, HYPER, "Hyperecho"),
		OPT_SELECT('F', enum seq_type, &seq, FLASH, "FLASH"),
		OPT_SELECT('S', enum seq_type, &seq, SPINECHO, "Spinecho"),
		OPT_SELECT('B', enum seq_type, &seq, BSSFP, "bSSFP"),
		OPT_FLOAT('1', &T1, "T1", "T1 [units of time]"),
		OPT_FLOAT('2', &T2, "T2", "T2 [units of time]"),
		OPT_FLOAT('b', &B1, "B1", "relative B1 [unitless]"),
		OPT_FLOAT('o', &offres, "OFF", "off-resonance [units of inverse time]"),
		OPT_FLOAT('r', &TR, "TR", "repetition time [units of time]"),
		OPT_FLOAT('e', &TE, "TE", "echo time [units of time]"),
		OPT_FLOAT('f', &FA, "FA", "flip angle [degrees]"),
		OPT_LONG('s', &SP, "SP", "spoiling (0: ideal, 1: conventional RF, 2: random RF)"),
		OPT_LONG('n',   &N, "N", "number of pulses"),
		OPT_LONG('u',   &unknowns, "U", "unknowns as bitmask (0: T1, 1: T2, 2: B1, 3: off-res)"),
		OPT_LONG('v',   &verbose, "V", "verbosity level"),
	};

	cmdline(&argc, argv, 1, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (2 < argc)
		save_states = true;
	
	if (3 < argc)
		save_sigder = true;

	if (4 < argc)
		save_statesder = true;

	long M;	
	if ((FMSSFP == seq) | (BSSFP == seq))
		M = 1;
	else if (FLASH == seq)
		M = N;
	else
		M = 2*N;
	
	complex float out_signal[N];
	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = N;

	// allocate on HEAP	to avoid memory limitation
	complex float (*out_states)[M][N];
	out_states = malloc(3 * sizeof *out_states);
    long dims_states[DIMS];
   	md_copy_dims(DIMS, dims_states, dims);
	dims_states[COEFF_DIM] = M;
	dims_states[COEFF2_DIM] = 3;

    complex float out_sigder[4][N];
    long dims_sigder[DIMS];
   	md_copy_dims(DIMS, dims_sigder, dims);
	dims_sigder[ITER_DIM] = 4;
   
	// allocate on HEAP	to avoid memory limitation
	complex float (*out_statesder)[3][M][N];
	out_statesder = malloc(4 * sizeof *out_statesder);
    long dims_statesder[DIMS];
   	md_copy_dims(DIMS, dims_statesder, dims_states);
	dims_statesder[ITER_DIM] = 4;

	complex float* signals = create_cfl(argv[1], DIMS, dims);

	complex float* states = (save_states ? create_cfl : anon_cfl)
								(save_states ? argv[2] : "", DIMS, dims_states);

	complex float* sigder = (save_sigder ? create_cfl : anon_cfl)
								(save_sigder ? argv[3] : "", DIMS, dims_sigder);

	complex float* statesder = (save_statesder ? create_cfl : anon_cfl)
								(save_statesder ? argv[4] : "", DIMS, dims_statesder);

	switch (seq) {

		case CPMG: cpmg_epg_der(N, M, out_signal, 
								   save_states ? out_states : NULL,
								   save_sigder ? out_sigder : NULL,
								   save_statesder ? out_statesder : NULL,
								   90.0, 180.0, TE, T1, T2, B1, offres); break;
		case FMSSFP: fmssfp_epg_der(N, M, out_signal,
								   save_states ? out_states : NULL,
								   save_sigder ? out_sigder : NULL,
								   save_statesder ? out_statesder : NULL,
								   FA, TR, T1, T2, B1, offres); break;

		case HYPER:  if (save_sigder || save_statesder)
							 error("No derivatives available for HYPER!");
						 else
							 hyperecho_epg(N, M, out_signal,
								   save_states ? out_states : NULL,
								   90.0, 180.0, TE, FA, T1, T2, B1, offres); break;
					
		case FLASH: flash_epg_der(N, M, out_signal,
								   save_states ? out_states : NULL,
								   save_sigder ? out_sigder : NULL,
								   save_statesder ? out_statesder : NULL,
								   FA, TR, T1, T2, B1, offres, SP); break;
	
		case SPINECHO:	 if (save_sigder || save_statesder)
							 error("No derivatives available for Spinecho!");
						 else
						 hahnspinecho_epg(N, M, out_signal, 
								   save_states ? out_states : NULL,
								   FA, TE, T1, T2, B1, offres); break;
		
		case BSSFP: bssfp_epg_der(N, M, out_signal,
								   save_states ? out_states : NULL,
								   save_sigder ? out_sigder : NULL,
								   save_statesder ? out_statesder : NULL,
								   FA, TR, T1, T2, B1, offres); break;

		default: error("sequence type not supported yet");
	}

	if (save_sigder) {

		if (0 < verbose) {

			int Q = bitcount(unknowns); // selected unknowns
			// determine indeces to unknowns in derivative
			unsigned long idx_unknowns[Q];
			getidxunknowns(Q, idx_unknowns, unknowns);
			
			int P = Q + 1; // selected unknowns + M0
			complex float fisher[P][P];
			float rCRB[P];

			compute_crb(P, rCRB, fisher, 4,  N, out_sigder, out_signal, idx_unknowns);
			normalize_crb(P, rCRB, N, TR, T1, T2, B1, offres, idx_unknowns);
			display_crb(P, rCRB, fisher, idx_unknowns);
		}

		// normalize selected derivatives	
		for (int n = 0; n < N; n++) {
			out_sigder[0][n] *= T1;
			out_sigder[1][n] *= T2;
			out_sigder[2][n] *= B1;
			/* out_sigder[4][n] *= 1; */
		}
	}

	long pos[DIMS] = {0};
	md_copy_block(DIMS, pos, dims, signals, dims, out_signal, CFL_SIZE);

	if (save_states)
		md_copy_block(DIMS, pos, dims_states, states, dims_states, out_states, CFL_SIZE);

	if (save_sigder)
		md_copy_block(DIMS, pos, dims_sigder, sigder, dims_sigder, out_sigder, CFL_SIZE);

	if (save_statesder)
		md_copy_block(DIMS, pos, dims_statesder, statesder, dims_statesder, out_statesder, CFL_SIZE);

	unmap_cfl(DIMS, dims, signals);
	unmap_cfl(DIMS, dims_states, states);
	unmap_cfl(DIMS, dims_sigder, sigder);
	unmap_cfl(DIMS, dims_statesder, statesder);

	return 0;
	}
