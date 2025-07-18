/* Copyright 2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/mri.h"

#include "moba/T1fun.h"
#include "moba/lorentzian.h"

#include "model_create.h"


const struct nlop_s* moba_get_nlop( struct nlop_data* data, const long map_dims[DIMS], const long out_dims[DIMS], const long param_dims[DIMS], const long enc_dims[DIMS], const complex float* enc)
	{
	const struct nlop_s* nlop = NULL;
	int n_params = param_dims[COEFF_DIM];
	switch (data->seq) {

	case IR_LL:

		if (n_params  != 3)
			error("Number of parameters does not match IR-LL model (Mss, M0, R1s)\n");

		nlop = nlop_T1_create(DIMS, map_dims, out_dims, param_dims, enc_dims, enc, 1, 1);
		break;

	case MPL:

		// M0 exists once, every pool has 3 parameters, we need at least one pool >3 parameters
		if ((n_params < 4) || ((n_params - 1) % 3 != 0))
			error("Number of parameters does not match MPL model\n");

		nlop = nlop_lorentzian_multi_pool_create(DIMS, out_dims, param_dims, enc_dims, enc);
		break;
	}

	return nlop;	
	}