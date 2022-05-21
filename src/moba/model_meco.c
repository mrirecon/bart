/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/debug.h"

#include "nlops/chain.h"
#include "nlops/nlop.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "noir/model.h"

#include "moba/model_meco.h"

#include "meco.h"

#include "simu/signals.h"

struct meco_s meco_create(const long dims[DIMS], const long y_dims[DIMS], const long x_dims[DIMS], const complex float* mask, const complex float* TE, const complex float* psf, enum meco_model sel_model, bool real_pd, enum fat_spec fat_spec, const float* scale_fB0, bool use_gpu, const struct noir_model_conf_s* conf)
{
	struct meco_s ret;

	if (MECO_PI == sel_model) {

		struct noir_s nlinv = noir_create(dims, mask, psf, conf);

		ret.nlop = nlinv.nlop;
		ret.linop = nlinv.linop;

	} else {

		// chain model
		struct nlop_s* meco = nlop_meco_create(DIMS, y_dims, x_dims, TE, sel_model, real_pd, fat_spec, scale_fB0, use_gpu);
		
		struct noir_s nlinv = noir_create3(dims, mask, psf, conf);

		const struct nlop_s* b = nlinv.nlop;
		nlinv.nlop = nlop_chain2(meco, 0, b, 0);
		nlop_free(b);

		auto c = nlinv.nlop;
		nlinv.nlop = nlop_permute_inputs(nlinv.nlop, 2, (const int[2]){1, 0});
		nlop_free(c);

		ret.nlop = nlop_flatten(nlinv.nlop);
		ret.linop = nlinv.linop;
		ret.linop_fB0 = meco_get_fB0_trafo(meco);
		ret.scaling = meco_get_scaling(meco);
		ret.weight_fB0_type = meco_get_weight_fB0_type(meco);

		nlop_free(meco);
		nlop_free(nlinv.nlop);		
	}

	return ret;
}

