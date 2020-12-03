/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/init.h"

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

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char usage_str[] = "<echo images> <TE> <parameters>";
static const char help_str[] = "Pixel-wise fitting";

int main_pixelfit(int argc, char* argv[])
{
	double start_time = timestamp();

	enum seq_type { BSSFP, FLASH, TSE, MOLLI, MGRE } seq = MGRE;

	unsigned int mgre_model = MECO_WFR2S;

	bool use_gpu = false;
	long patch_size[2] = { 1, 1 };

	unsigned int iter = 4;

	const struct opt_s opts[] = {

		OPT_SELECT('F', enum seq_type, &seq, FLASH, "FLASH"),
		OPT_SELECT('B', enum seq_type, &seq, BSSFP, "bSSFP"),
		OPT_SELECT('T', enum seq_type, &seq, TSE, "TSE"),
		OPT_SELECT('M', enum seq_type, &seq, MOLLI, "MOLLI"),
		OPT_SELECT('G', enum seq_type, &seq, MGRE, "MGRE"),
		OPT_UINT('m', &mgre_model, "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF } [default: WFR2S]"),
		OPT_UINT('i', &iter, "iter", "Number of Newton steps"),
		OPT_VEC2('p', &patch_size, "px,py", "(patch size)"),
		OPT_SET('g', &use_gpu, "use gpu"),
	};

	cmdline(&argc, argv, 2, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	long y_dims[DIMS];
	complex float* y = load_cfl(argv[1], DIMS, y_dims);

	long TE_dims[DIMS];
	complex float* TE = load_cfl(argv[2], DIMS, TE_dims);

	assert(y_dims[TE_DIM] == TE_dims[TE_DIM]);

	long x_dims[DIMS];
	md_select_dims(DIMS, ~TE_FLAG, x_dims, y_dims);
	x_dims[COEFF_DIM] = set_num_of_coeff(mgre_model);

	complex float* x = create_cfl(argv[3], DIMS, x_dims);

	md_zfill(DIMS, x_dims, x, 0.);


	long y_patch_dims[DIMS];
	md_select_dims(DIMS, TE_FLAG, y_patch_dims, y_dims);
	y_patch_dims[0] = patch_size[0];
	y_patch_dims[1] = patch_size[1];

	long x_patch_dims[DIMS];
	md_select_dims(DIMS, COEFF_FLAG, x_patch_dims, x_dims);
	x_patch_dims[0] = patch_size[0];
	x_patch_dims[1] = patch_size[1];


	// create signal model
	struct nlop_s* nlop = NULL;

	switch (seq) {

	case FLASH: break;
	case BSSFP: break;
	case TSE:   break;
	case MOLLI: break;
	case MGRE:  
	{
		float scale_fB0[2] = { 0., 1. };
		nlop = nlop_meco_create(DIMS, y_patch_dims, x_patch_dims, TE, mgre_model, false, scale_fB0, use_gpu);
		break;
	}

	default: error("sequence type not supported");

	}

	struct iter_admm_conf admm_conf = iter_admm_defaults;
	admm_conf.rho = 1.E-5;

	struct lsqr_conf lsqr_conf = lsqr_defaults;
	lsqr_conf.it_gpu = false;
	lsqr_conf.warmstart = true;

	NESTED(void, lsqr_cont, (iter_conf* iconf))
	{
		auto aconf = CAST_DOWN(iter_admm_conf, iconf);

		aconf->maxiter = MIN(admm_conf.maxiter, 10. * powf(2., logf(1. / iconf->alpha)));
		aconf->cg_eps = admm_conf.cg_eps * iconf->alpha;
	};

	lsqr_conf.icont = lsqr_cont;


	const struct operator_p_s* lsqr = lsqr2_create(&lsqr_conf, iter2_admm, CAST_UP(&admm_conf), NULL, &nlop->derivative[0][0], NULL, 0, NULL, NULL, NULL);


	struct iter3_irgnm_conf irgnm_conf = iter3_irgnm_defaults;
	irgnm_conf.iter = iter;


	complex float* y_patch = md_alloc(DIMS, y_patch_dims, CFL_SIZE);
	complex float* x_patch = md_alloc(DIMS, x_patch_dims, CFL_SIZE);


	long pos[DIMS] = { 0 };

	do {

		md_copy_block(DIMS, pos, y_patch_dims, y_patch, y_dims, y, CFL_SIZE);
		md_copy_block(DIMS, pos, x_patch_dims, x_patch, x_dims, x, CFL_SIZE);

		iter4_irgnm2(CAST_UP(&irgnm_conf), nlop,
				2 * md_calc_size(DIMS, x_patch_dims), (float*)x_patch, NULL,
				2 * md_calc_size(DIMS, y_patch_dims), (const float*)y_patch, lsqr,
				(struct iter_op_s){ NULL, NULL });

		md_copy_block(DIMS, pos, x_dims, x, x_patch_dims, x_patch, CFL_SIZE);

	} while(md_next(DIMS, y_dims, ~TE_FLAG, pos));


	operator_p_free(lsqr);
	nlop_free(nlop);

	unmap_cfl(DIMS, y_dims, y);
	unmap_cfl(DIMS, TE_dims, TE);
	unmap_cfl(DIMS, x_dims, x);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total Time: %.2f s\n", recosecs);

	exit(0);
}
