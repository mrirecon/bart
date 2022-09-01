/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "noir/model.h"

#include "moba/blochfun.h"
#include "moba/T1phyfun.h"
#include "moba/moba.h"

#include "model_moba.h"


struct mobamod moba_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* b1,
		const complex float* psf, const struct noir_model_conf_s* conf, struct moba_conf_s* data, _Bool use_gpu)
{
	long data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_s nlinv = noir_create3(data_dims, mask, psf, conf);
	struct mobamod ret;

	// FIXME: unify them more
	long der_dims[DIMS];
	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
        long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|COEFF_FLAG|TIME2_FLAG, der_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TIME_FLAG|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, in_dims, dims);
        md_select_dims(DIMS, TE_FLAG|TIME_FLAG|TIME2_FLAG, TI_dims, dims);

	struct nlop_s* model = NULL;

	switch (data->model) {

	case MDB_T1:
	case MDB_T2:
	case MDB_MGRE:

                // FIXME: Integrate other models here
		assert(0);
		break;

        case MDB_T1_PHY:

		model = nlop_T1_phy_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, use_gpu);
		break;

	case MDB_BLOCH:

		for (int i = 0; i < 4; i++)
			debug_printf(DP_DEBUG2, "FP Scale[%d]=%f\n", i, crealf(data->other.scale[i]));

                // Turn off matching of T2 for IR FLASH

		if (SEQ_IRFLASH == data->sim.seq.seq_type)
			data->other.scale[2] = 0.;

		model = nlop_bloch_create(DIMS, der_dims, map_dims, out_dims, in_dims, b1, data, use_gpu);
		break;
	}

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(model, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(model, 0)->dims);

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 1)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(nlinv.nlop, 0)->dims);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(model, 0, b, 0);
	nlop_free(b);

	nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });

	nlop_free(c);

	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;

	if (MDB_BLOCH == data->model)
                ret.linop_alpha = bloch_get_alpha_trafo(model);
        else if (MDB_T1_PHY == data->model)
                ret.linop_alpha = T1_get_alpha_trafo(model);

	nlop_free(nlinv.nlop);

	return ret;
}


