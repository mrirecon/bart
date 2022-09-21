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

#include "linops/someops.h"
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

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nlop_jacobian.h"

#include "moba/meco.h"
#include "moba/exp.h"
#include "moba/T1fun.h"

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


	enum seq_type { BSSFP, FLASH, TSE, MOLLI, MGRE, DIFF, IR_LL } seq = MGRE;

	unsigned int mgre_model = MECO_WFR2S;

	bool use_gpu = false;

	unsigned int iter = 5;

	const char* basis_file = NULL;

	const struct opt_s opts[] = {

#if 0
		OPT_SELECT('F', enum seq_type, &seq, FLASH, "FLASH"),
		OPT_SELECT('B', enum seq_type, &seq, BSSFP, "bSSFP"),
		OPT_SELECT('M', enum seq_type, &seq, MOLLI, "MOLLI"),
#endif
		OPT_SELECT('T', enum seq_type, &seq, TSE, "TSE"),
		OPT_SELECT('L', enum seq_type, &seq, IR_LL, "Inversion Recovery Look-Locker"),
		OPT_SELECT('G', enum seq_type, &seq, MGRE, "MGRE"),
		OPT_SELECT('D', enum seq_type, &seq, DIFF, "diffusion"),
		OPT_UINT('m', &mgre_model, "model", "Select the MGRE model from enum { WF = 0, WFR2S, WF2R2S, R2S, PHASEDIFF } [default: WFR2S]"),
		OPT_UINT('i', &iter, "iter", "Number of IRGNM steps"),
		OPT_SET('g', &use_gpu, "use gpu"),
		OPT_INFILE('B', &basis_file, "file", "temporal (or other) basis"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

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

	default:

		error("sequence type not supported");
	}


	complex float* x = create_cfl(coeff_file, DIMS, x_dims);

	md_clear(DIMS, x_dims, x, CFL_SIZE);



	long y_patch_dims[DIMS];
	long x_patch_dims[DIMS];
	long y_patch_sig_dims[DIMS];

	md_select_dims(DIMS, FFT_FLAGS | TE_FLAG | COEFF_FLAG, y_patch_dims, y_dims);
	md_select_dims(DIMS, FFT_FLAGS | TE_FLAG | COEFF_FLAG, y_patch_sig_dims, y_sig_dims);
	md_select_dims(DIMS, FFT_FLAGS | TE_FLAG | COEFF_FLAG, x_patch_dims, x_dims);


	// create signal model
	struct nlop_s* nlop = NULL;

	switch (seq) {
	
	case IR_LL: ;

		long map_dims[DIMS];
		md_select_dims(DIMS, ~(COEFF_FLAG | TE_FLAG), map_dims, x_patch_dims);
		nlop = nlop_T1_create(DIMS, map_dims, y_patch_sig_dims, x_patch_dims, enc_dims, enc, 1, 1);
		
		if (NULL != basis) {

			long max_dims[DIMS];
			md_max_dims(DIMS, ~0, max_dims, bas_dims, y_patch_sig_dims);

			unsigned long oflags = ~md_nontriv_dims(DIMS, y_patch_dims);
			unsigned long iflags = ~md_nontriv_dims(DIMS, y_patch_sig_dims);
			unsigned long bflags = ~md_nontriv_dims(DIMS, bas_dims);
			const struct nlop_s* nlop_bas = nlop_from_linop_F(linop_fmac_create(DIMS, max_dims, oflags, iflags, bflags, basis));
			nlop = nlop_chain_FF(nlop, nlop_bas);

			long tdims[DIMS];
			md_transpose_dims(DIMS, 5, 6, tdims, y_patch_dims);
			nlop = (struct nlop_s*)nlop_reshape_out_F(nlop, 0, DIMS, tdims);
			nlop = nlop_zrprecomp_jacobian_F(nlop);
			nlop = (struct nlop_s*)nlop_reshape_out_F(nlop, 0, DIMS, y_patch_dims);

			auto tmp = linop_stack_FF(6, 6, linop_identity_create(DIMS, map_dims), linop_identity_create(DIMS, map_dims));
			tmp = linop_stack_FF(6, 6, tmp, linop_zreal_create(DIMS, map_dims));
			nlop = nlop_chain_FF(nlop_from_linop_F(tmp), nlop);
		}

		md_zfill(DIMS, x_dims, x, 1.);

		break;

	case MGRE: ;

		float scale_fB0[2] = { 0., 1. };
		assert(md_check_equal_dims(DIMS, y_patch_dims, y_patch_sig_dims, ~0));
		nlop = nlop_meco_create(DIMS, y_patch_dims, x_patch_dims, enc, mgre_model, false, FAT_SPEC_1, scale_fB0, use_gpu);
		break;

	case TSE:
	case DIFF: ;

		assert(md_check_equal_dims(DIMS, y_patch_dims, y_patch_sig_dims, ~0));
		long dims[DIMS];
		md_copy_dims(DIMS, dims, y_patch_dims);
		dims[COEFF_DIM] = enc_dims[COEFF_DIM];

		auto nl = nlop_exp_create(DIMS, dims, enc);
		nlop = nlop_flatten(nl);
		nlop_free(nl);
		break;

	default:
		__builtin_unreachable();
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

	if (use_gpu) {

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

	} while(md_next(DIMS, y_dims, ~(FFT_FLAGS | TE_FLAG | COEFF_FLAG), pos));

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

