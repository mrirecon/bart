/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2022. Institute of Biomedical Imaging. TU Graz.
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

#include "nlops/nlop.h"

#include "moba/meco.h"
#include "moba/exp.h"

#include "simu/signals.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Pixel-wise fitting of physical signal models.";

int main_mobafit(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* enc_file = NULL;
	const char* echo_file = NULL;
	const char* coeff_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &enc_file, "enc"),
		ARG_INFILE(true, &echo_file, "echo/contrast images"),
		ARG_OUTFILE(false, &coeff_file, "coefficients"),
	};


	enum seq_type { BSSFP, FLASH, TSE, MOLLI, MGRE, DIFF } seq = MGRE;

	unsigned int mgre_model = MECO_WFR2S;

	bool use_gpu = false;
	long patch_size[3] = { 1, 1, 1 };

	unsigned int iter = 5;

	const struct opt_s opts[] = {

#if 0
		OPT_SELECT('F', enum seq_type, &seq, FLASH, "FLASH"),
		OPT_SELECT('B', enum seq_type, &seq, BSSFP, "bSSFP"),
		OPT_SELECT('M', enum seq_type, &seq, MOLLI, "MOLLI"),
#endif
		OPT_SELECT('T', enum seq_type, &seq, TSE, "TSE"),
		OPT_SELECT('G', enum seq_type, &seq, MGRE, "MGRE"),
		OPT_SELECT('D', enum seq_type, &seq, DIFF, "diffusion"),
		OPT_UINT('m', &mgre_model, "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF } [default: WFR2S]"),
		OPT_UINT('i', &iter, "iter", "Number of IRGNM steps"),
		OPT_VEC3('p', &patch_size, "px,py,pz", "(patch size)"),
		OPT_SET('g', &use_gpu, "use gpu"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	long enc_dims[DIMS];
	complex float* enc = load_cfl(enc_file, DIMS, enc_dims);

	long y_dims[DIMS];
	complex float* y = load_cfl(echo_file, DIMS, y_dims);

	assert(y_dims[TE_DIM] == enc_dims[TE_DIM]);

	long x_dims[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, x_dims, y_dims);

	switch (seq) {

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

	default:

		error("sequence type not supported");
	}


	complex float* x = create_cfl(coeff_file, DIMS, x_dims);

	md_clear(DIMS, x_dims, x, CFL_SIZE);



	long y_patch_dims[DIMS];
	md_select_dims(DIMS, TE_FLAG, y_patch_dims, y_dims);
	md_copy_dims(3, y_patch_dims, patch_size);

	long x_patch_dims[DIMS];
	md_select_dims(DIMS, COEFF_FLAG, x_patch_dims, x_dims);
	md_copy_dims(3, x_patch_dims, patch_size);


	// create signal model
	struct nlop_s* nlop = NULL;

	switch (seq) {

	case MGRE: ;

		float scale_fB0[2] = { 0., 1. };
		nlop = nlop_meco_create(DIMS, y_patch_dims, x_patch_dims, enc, mgre_model, false, FAT_SPEC_1, scale_fB0, use_gpu);
		break;

	case TSE:
	case DIFF: ;

		long dims[DIMS];
		md_copy_dims(DIMS, dims, y_patch_dims);
		dims[COEFF_DIM] = enc_dims[COEFF_DIM];

		auto nl = nlop_exp_create(DIMS, dims, enc);
		nlop = nlop_flatten(nl);
		nlop_free(nl);
		break;

	default: ;
	}



	struct iter_conjgrad_conf conjgrad_conf = iter_conjgrad_defaults;
	struct lsqr_conf lsqr_conf = lsqr_defaults;
	lsqr_conf.it_gpu = false;

	const struct operator_p_s* lsqr = lsqr2_create(&lsqr_conf, iter2_conjgrad, CAST_UP(&conjgrad_conf), NULL, &nlop->derivative[0][0], NULL, 0, NULL, NULL, NULL);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	irgnm_conf.iter = iter;


	complex float* y_patch = md_alloc(DIMS, y_patch_dims, CFL_SIZE);
	complex float* x_patch = md_alloc(DIMS, x_patch_dims, CFL_SIZE);


	long pos[DIMS] = { 0 };

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
				(struct iter_op_s){ NULL, NULL });

		md_copy_block(DIMS, pos, x_dims, x, x_patch_dims, x_patch, CFL_SIZE);

	} while(md_next(DIMS, y_dims, ~TE_FLAG, pos));

	md_free(x_patch);
	md_free(y_patch);


	operator_p_free(lsqr);
	nlop_free(nlop);

	unmap_cfl(DIMS, y_dims, y);
	unmap_cfl(DIMS, enc_dims, enc);
	unmap_cfl(DIMS, x_dims, x);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);

	return 0;
}

