/* Copyright 2018-2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2020 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "noir/model.h"

#include "moba/T2fun.h"

#include "model_T2.h"




struct mobamod T2_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf, const struct noir_model_conf_s* conf, _Bool use_gpu)
{
	long data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_s nlinv = noir_create3(data_dims, mask, psf, conf);
	struct mobamod ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG, TI_dims, dims);

	// chain T2 model
	struct nlop_s* T2 = nlop_T2_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, use_gpu);


	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(T2, 0, b, 0);
	nlop_free(b);

	nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });
	nlop_free(c);

	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;

	nlop_free(nlinv.nlop);
	return ret;
}


