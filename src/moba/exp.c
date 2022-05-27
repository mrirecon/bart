/* Copyright 2022. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "num/multind.h"

#include "linops/fmac.h"

#include "nlops/tenmul.h"
#include "nlops/zexp.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"

#include "misc/mri.h"

#include "exp.h"


struct nlop_s* nlop_exp_create(int N, const long dims[N], const complex float* enc)
{
	auto lo = linop_fmac_create(N, dims, COEFF_FLAG, TE_FLAG, FFT_FLAGS,  enc);

	long out_dims[N];
	md_select_dims(N, FFT_FLAGS | TE_FLAG, out_dims, dims);

	auto nl1 = nlop_from_linop_F(lo);
	auto nl2 = nlop_zexp_create(N, out_dims);
	auto nl3 = nlop_chain_FF(nl1, nl2);

	long dims1[N];
	md_select_dims(N, ~TE_FLAG, dims1, out_dims);

	auto nl4 = nlop_tenmul_create(N, out_dims, dims1, out_dims);
	auto nl5 = nlop_chain2_FF(nl3, 0, nl4, 1);

	return nl5;
}

