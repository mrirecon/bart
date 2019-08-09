/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
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

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "noir/model.h"

#include "moba/T1fun.h"

#include "model_T1.h"




struct T1_s T1_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf, const struct noir_model_conf_s* conf, _Bool use_gpu)
{
	struct noir_s nlinv = noir_create3(dims, mask, psf, conf);
	struct T1_s ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME2_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG|TIME2_FLAG, TI_dims, dims);

	in_dims[COEFF_DIM] = 3;

#if 1
	// chain T1 model
	struct nlop_s* T1 = nlop_T1_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, use_gpu);

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(T1, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(T1, 0)->dims);

	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 0)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_domain(nlinv.nlop, 1)->dims);
	debug_print_dims(DP_INFO, DIMS, nlop_generic_codomain(nlinv.nlop, 0)->dims);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2(T1, 0, b, 0);
	nlop_free(b);

	nlinv.nlop = nlop_permute_inputs(c, 2, (const int[2]){ 1, 0 });
	nlop_free(c);
#endif

	ret.nlop = nlop_flatten(nlinv.nlop);
	ret.linop = nlinv.linop;
	nlop_free(nlinv.nlop);

	return ret;
}


