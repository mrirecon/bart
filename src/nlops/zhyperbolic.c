/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "nlops/nlop_jacobian.h"

#include "zhyperbolic.h"


struct zsinh_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zsinh_s);

static void zsinh_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zsinh_apply(const nlop_data_t* _data, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{
	UNUSED(_data);

	if (NULL != der)
		md_zcosh(N, dims, der, src);

	md_zsinh(N, dims, dst, src);
}

struct nlop_s* nlop_zsinh_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zsinh_s, data);
	SET_TYPEID(zsinh_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zsinh_apply, zsinh_free);
}






struct zcosh_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(zcosh_s);

static void zcosh_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zcosh_apply(const nlop_data_t* _data, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{
	UNUSED(_data);

	if (NULL != der)
		md_zsinh(N, dims, der, src);

	md_zcosh(N, dims, dst, src);
}

struct nlop_s* nlop_zcosh_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zcosh_s, data);
	SET_TYPEID(zcosh_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zcosh_apply, zcosh_free);
}
