/* Copyright 2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "misc/debug.h"

#include "misc/misc.h"
#include "misc/mri.h"

#include "moba/T1fun.h"
#include "moba/lorentzian.h"
#include "moba/exp.h"

#include "model_create.h"



const struct nlop_s* moba_get_nlop( struct nlop_data* data, const long map_dims[DIMS], const long out_dims[DIMS], const long param_dims[DIMS], const long enc_dims[DIMS], complex float* enc)
	{
	const struct nlop_s* nlop = NULL;
	int n_params = param_dims[COEFF_DIM];

	long dims[DIMS];
	md_copy_dims(DIMS, dims, out_dims);
	dims[COEFF_DIM] = enc_dims[COEFF_DIM];

	if (data->seq == TSE)
		md_zsmul(DIMS, enc_dims, enc, enc, -1.);

	switch (data->seq) {

	case IR:

		if (n_params  != 3)
			error("Number of parameters does not match IR model (M0, R1, c)\n");

		nlop = nlop_ir_create(DIMS, out_dims, enc);
		break;
		
	case IR_LL:

		if (n_params  != 3)
			error("Number of parameters does not match IR-LL model (Mss, M0, R1s)\n");

		nlop = nlop_T1_create(DIMS, map_dims, out_dims, param_dims, enc_dims, enc, 1, 1);
		break;

	case TSE:
	case DIFF:

		nlop = nlop_exp_create(DIMS, dims, enc);
		break;

	case MPL:
		// M0 exists once, every pool has 3 parameters, we need at least one pool >3 parameters
		if ((n_params < 4) || ((n_params - 1) % 3 != 0))
			error("Number of parameters does not match MPL model\n");

		nlop = nlop_lorentzian_multi_pool_create(DIMS, out_dims, param_dims, enc_dims, enc);
		break;

	default:
	debug_printf(DP_DEBUG2, "Sequence Type %c \n", data->seq);

	error("sequence type not supported\n");
	}

	return nlop;	
	}