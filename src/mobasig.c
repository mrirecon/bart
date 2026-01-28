/* Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/mri.h"

#include "nlops/nlop.h"

#include "moba/model_moba.h"
#include "moba/T1fun.h"
#include "moba/lorentzian.h"


static const char help_str[] = "Forward calculation of pysical signal models.";


int main_mobasig(int argc, char* argv[argc])
{
	const char* param_file = NULL;
	const char* signal_file = NULL;
	const char* enc_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &param_file, "parameters/coefficients"),
		ARG_INFILE(true, &enc_file, "encoding"),
		ARG_OUTFILE(true, &signal_file, "signal"),
	};

	struct mobafit_model_config data;
	data.seq = IR_LL;
	data.mgre_model = MECO_WFR2S;

	const struct opt_s opts[] = {

		OPT_SELECT('I', enum seq_type, &(data.seq), IR, "Inversion Recovery: f(M0, R1, c) =  M0 * (1 - exp(-t * R1 + c))"),
		OPT_SELECT('L', enum seq_type, &(data.seq), IR_LL, "Inversion Recovery Look-Locker (M0', MSS, R1S)"),
		OPT_SELECT('M', enum seq_type, &(data.seq), MPL, "Multi-Pool-Lorentzian model"),
		OPT_SELECT('D', enum seq_type, &(data.seq), DIFF, "diffusion"),
		OPT_SELECT('T', enum seq_type, &(data.seq), TSE, "Multi-Echo Spin Echo: f(M0, R2) = M0 * exp(-t * R2)"),
		OPT_SELECT('G', enum seq_type, &(data.seq), MGRE, "MGRE"),
		OPT_PINT('m',  (int*)&(data.mgre_model), "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF } [default: WFR2S]"),

	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long param_dims[DIMS];
	complex float* coeff_data = load_cfl(param_file, DIMS, param_dims);

	long enc_dims[DIMS];
	complex float* enc = load_cfl(enc_file, DIMS, enc_dims);

	long out_dims[DIMS];
	md_select_dims(DIMS, ~(TE_FLAG | COEFF_FLAG), out_dims, param_dims);
	out_dims[TE_DIM] = enc_dims[TE_DIM];
	complex float* sig_data = create_cfl(signal_file, DIMS, out_dims);


	const struct nlop_s* nlop = moba_get_nlop(&data, out_dims, param_dims, enc_dims, enc);

	nlop_apply(nlop, DIMS, out_dims, sig_data, DIMS, param_dims, coeff_data);
	nlop_free(nlop);

	unmap_cfl(DIMS, param_dims, coeff_data);
	unmap_cfl(DIMS, out_dims, sig_data);
	unmap_cfl(DIMS, enc_dims, enc);

	return 0;
}
