/* Copyright 2019-2021. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2022-2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/stack.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "noir/model.h"

#include "moba/T1fun.h"

#include "model_T1.h"




struct mobamod T1_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf,
				float scaling_M0, float scaling_R1s, const struct noir_model_conf_s* conf, float fov)
{
	long data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_s nlinv = noir_create(data_dims, mask, psf, conf);
	struct mobamod ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TIME_FLAG|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG|TIME_FLAG|TIME2_FLAG, TI_dims, dims);

	// chain T1 model

	struct nlop_s* T1;

	if (conf->noncart) { // overgridding with factor two

		long map_dims2[DIMS];
		long out_dims2[DIMS];
		long in_dims2[DIMS];

		md_copy_dims(DIMS, map_dims2, map_dims);
		md_copy_dims(DIMS, out_dims2, out_dims);
		md_copy_dims(DIMS, in_dims2, in_dims);

		long red_fov[3];

		for (int i = 0; i < 3; i++)
			red_fov[i] = (1 == map_dims[i]) ? 1 : (map_dims[i] * fov);

		if (1. != fov) {

			md_copy_dims(3, map_dims2, red_fov);
			md_copy_dims(3, out_dims2, red_fov);
			md_copy_dims(3, in_dims2, red_fov);
		}

		T1 = nlop_T1_create(DIMS, map_dims2, out_dims2, in_dims2, TI_dims, TI, scaling_M0, scaling_R1s);

		T1 = nlop_chain_FF(T1, nlop_from_linop_F(linop_resize_center_create(DIMS, out_dims, out_dims2)));
		T1 = nlop_chain_FF(nlop_from_linop_F(linop_resize_center_create(DIMS, in_dims2, in_dims)), T1);

	} else {

		T1 = nlop_T1_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, scaling_M0, scaling_R1s);
	}

	debug_printf(DP_INFO, "T1 Model created:\n Model ");
	nlop_debug(DP_INFO, T1);

	debug_printf(DP_INFO, "NLINV ");
	nlop_debug(DP_INFO, nlinv.nlop);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2_FF(T1, 0, b, 0);

	nlinv.nlop = nlop_permute_inputs_F(c, 2, (const int[2]){ 1, 0 });
	ret.nlop = nlop_flatten_inputs_F(nlinv.nlop);
	ret.linop = nlinv.linop;

	return ret;
}


