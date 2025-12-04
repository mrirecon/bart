/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "stdbool.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "linops/someops.h"
#include "linops/fmac.h"

#include "nlops/snlop.h"
#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/someops.h"
#include "nlops/tenmul.h"
#include "nlops/zexp.h"
#include "nlops/ztrigon.h"
#include "nlops/zhyperbolic.h"
#include "nlops/stack.h"

#include "smath.h"

typedef const struct nlop_s* (*nlop_diag_create_t)(int N, const long[N]);

static arg_t snlop_diag_append(arg_t arg, nlop_diag_create_t create, bool keep)
{
	const struct iovec_s* iov = arg_get_iov(arg);
	const struct nlop_s* nlop = create(iov->N, iov->dims);

	return snlop_append_nlop_F(arg, nlop, keep);
}

typedef struct linop_s* (*linop_diag_create_t)(int N, const long[N]);

static arg_t snlop_linop_diag_append(arg_t arg, linop_diag_create_t create, bool keep)
{
	const struct iovec_s* iov = arg_get_iov(arg);
	const struct nlop_s* nlop = nlop_from_linop_F(create(iov->N, iov->dims));

	return snlop_append_nlop_F(arg, nlop, keep);
}

arg_t snlop_abs(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zabs_create, true);
}

arg_t snlop_abs_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zabs_create, false);
}

arg_t snlop_exp(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zexp_create, true);
}

arg_t snlop_exp_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zexp_create, false);
}

arg_t snlop_log(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zlog_create, true);
}

arg_t snlop_log_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zlog_create, false);
}


arg_t snlop_cos(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zcos_create, true);
}

arg_t snlop_cos_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zcos_create, false);
}

arg_t snlop_sin(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zsin_create, true);
}

arg_t snlop_sin_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zsin_create, false);
}


arg_t snlop_cosh(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zcosh_create, true);
}

arg_t snlop_cosh_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zcosh_create, false);
}

arg_t snlop_sinh(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zsinh_create, true);
}

arg_t snlop_sinh_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zsinh_create, false);
}

arg_t snlop_sqrt(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zsqrt_create, true);
}

arg_t snlop_sqrt_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zsqrt_create, false);
}

arg_t snlop_inv(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zinv_create, true);
}

arg_t snlop_inv_F(arg_t arg)
{
	return snlop_diag_append(arg, nlop_zinv_create, false);
}

arg_t snlop_real(arg_t arg)
{
	return snlop_linop_diag_append(arg, linop_zreal_create, true);
}

arg_t snlop_real_F(arg_t arg)
{
	return snlop_linop_diag_append(arg, linop_zreal_create, false);
}

arg_t snlop_conj(arg_t arg)
{
	return snlop_linop_diag_append(arg, linop_zconj_create, true);
}

arg_t snlop_conj_F(arg_t arg)
{
	return snlop_linop_diag_append(arg, linop_zconj_create, false);
}



arg_t snlop_spow(arg_t arg, complex float pow)
{
	const struct iovec_s* iov = arg_get_iov(arg);
	const struct nlop_s* nlop = nlop_zspow_create(iov->N, iov->dims, pow);

	return snlop_append_nlop_F(arg, nlop, true);
}

arg_t snlop_spow_F(arg_t arg, complex float pow)
{
	const struct iovec_s* iov = arg_get_iov(arg);
	const struct nlop_s* nlop = nlop_zspow_create(iov->N, iov->dims, pow);

	return snlop_append_nlop_F(arg, nlop, false);
}


arg_t snlop_cdiag(arg_t arg, int _N, const long dims[_N], const complex float* diag)
{
	const struct iovec_s* iov = arg_get_iov(arg);

	int N = MAX(_N, iov->N);

	long ndims[N];
	md_singleton_dims(N, ndims);
	md_copy_dims(_N, ndims, dims);

	assert(! (md_nontriv_dims(N, ndims) & (~md_nontriv_dims(iov->N, iov->dims))));
	assert(md_check_compat(iov->N, ~0UL, ndims, iov->dims));

	const struct nlop_s* nlop = nlop_from_linop_F(linop_cdiag_create(N, iov->dims, md_nontriv_dims(N, iov->dims), diag));
	nlop = nlop_reshape_in_F(nlop, 0, _N, dims);

	return snlop_append_nlop_F(arg, nlop, true);
}

arg_t snlop_cdiag_F(arg_t arg, int _N, const long dims[_N], const complex float* diag)
{
	const struct iovec_s* iov = arg_get_iov(arg);

	int N = MAX(_N, iov->N);

	long ndims[N];
	md_singleton_dims(N, ndims);
	md_copy_dims(_N, ndims, dims);

	assert(! (md_nontriv_dims(N, ndims) & (~md_nontriv_dims(iov->N, iov->dims))));
	assert(md_check_compat(iov->N, ~0UL, ndims, iov->dims));

	const struct nlop_s* nlop = nlop_from_linop_F(linop_cdiag_create(N, iov->dims, md_nontriv_dims(N, iov->dims), diag));
	nlop = nlop_reshape_in_F(nlop, 0, _N, dims);

	return snlop_append_nlop_F(arg, nlop, false);
}

arg_t snlop_scale(arg_t arg, complex float scale)
{
	const struct iovec_s* iov = arg_get_iov(arg);

	const struct nlop_s* nlop = nlop_from_linop_F(linop_scale_create(iov->N, iov->dims, scale));

	return snlop_append_nlop_F(arg, nlop, true);
}

arg_t snlop_scale_F(arg_t arg, complex float scale)
{
	const struct iovec_s* iov = arg_get_iov(arg);

	const struct nlop_s* nlop = nlop_from_linop_F(linop_scale_create(iov->N, iov->dims, scale));

	return snlop_append_nlop_F(arg, nlop, false);
}


arg_t snlop_fmac(arg_t arg, int _N, const long dims[_N], const complex float* ten, unsigned long oflags)
{
	const struct iovec_s* iov = arg_get_iov(arg);

	int N = MAX(_N, iov->N);

	long mdims[N];
	md_singleton_dims(N, mdims);
	md_copy_dims(_N, mdims, dims);

	assert(md_check_compat(iov->N, ~0UL, mdims, iov->dims));

	md_max_dims(iov->N, ~0UL, mdims, mdims, iov->dims);

	unsigned long iflags = ~md_nontriv_dims(iov->N, iov->dims);
	unsigned long tflags = ~md_nontriv_dims(_N, dims);

	const struct nlop_s* nlop = nlop_from_linop_F(linop_fmac_create(N, mdims, oflags, iflags, tflags, ten));
	nlop = nlop_reshape_in_F(nlop, 0, iov->N, iov->dims);

	return snlop_append_nlop_F(arg, nlop, true);
}

arg_t snlop_fmac_F(arg_t arg, int _N, const long dims[_N], const complex float* ten, unsigned long oflags)
{
	const struct iovec_s* iov = arg_get_iov(arg);

	int N = MAX(_N, iov->N);

	long mdims[N];
	md_singleton_dims(N, mdims);
	md_copy_dims(_N, mdims, dims);

	assert(md_check_compat(iov->N, ~0UL, mdims, iov->dims));

	md_max_dims(N, ~0UL, mdims, mdims, iov->dims);

	unsigned long iflags = ~md_nontriv_dims(iov->N, iov->dims);
	unsigned long tflags = ~md_nontriv_dims(_N, dims);

	const struct nlop_s* nlop = nlop_from_linop_F(linop_fmac_create(N, mdims, oflags, iflags, tflags, ten));
	nlop = nlop_reshape_in_F(nlop, 0, iov->N, iov->dims);

	return snlop_append_nlop_F(arg, nlop, false);
}



arg_t snlop_mul(arg_t a, arg_t b, unsigned long flags)
{
	const struct iovec_s* iova = arg_get_iov(a);
	const struct iovec_s* iovb = arg_get_iov(b);

	int N = MAX(iova->N, iovb->N);
	long adims[N];
	long bdims[N];
	long mdims[N];

	md_singleton_dims(N, adims);
	md_singleton_dims(N, bdims);

	md_copy_dims(iova->N, adims, iova->dims);
	md_copy_dims(iovb->N, bdims, iovb->dims);

	md_max_dims(N, ~0UL, mdims, adims, bdims);
	md_select_dims(N, ~flags, mdims, mdims);

	const struct nlop_s* nlop = nlop_tenmul_create(N, mdims, adims, bdims);

	nlop = nlop_reshape_in_F(nlop, 0, iova->N, iova->dims);
	nlop = nlop_reshape_in_F(nlop, 1, iovb->N, iovb->dims);

	return snlop_append_nlop_generic_F(2, (arg_t[2]){ a, b }, nlop, true);
}

arg_t snlop_mul_F(arg_t a, arg_t b, unsigned long flags)
{
	const struct iovec_s* iova = arg_get_iov(a);
	const struct iovec_s* iovb = arg_get_iov(b);

	int N = MAX(iova->N, iovb->N);
	long adims[N];
	long bdims[N];
	long mdims[N];

	md_singleton_dims(N, adims);
	md_singleton_dims(N, bdims);

	md_copy_dims(iova->N, adims, iova->dims);
	md_copy_dims(iovb->N, bdims, iovb->dims);

	md_max_dims(N, ~0UL, mdims, adims, bdims);
	md_select_dims(N, ~flags, mdims, mdims);

	const struct nlop_s* nlop = nlop_tenmul_create(N, mdims, adims, bdims);

	nlop = nlop_reshape_in_F(nlop, 0, iova->N, iova->dims);
	nlop = nlop_reshape_in_F(nlop, 1, iovb->N, iovb->dims);

	return snlop_append_nlop_generic_F(2, (arg_t[2]){ a, b }, nlop, false);
}

arg_t snlop_div(arg_t a, arg_t b, unsigned long flags)
{
	assert(arg_check(a));
	assert(arg_check(b));

	arg_t inv = snlop_inv(b);
	arg_t ret = snlop_mul(a, inv, flags);
	assert(arg_check(a));
	assert(arg_check(b));

	snlop_del_arg(inv);

	assert(arg_check(a));
	assert(arg_check(b));

	return ret;
}

arg_t snlop_div_simple(arg_t a, arg_t b)
{
	return snlop_div(a, b, 0);
}

arg_t snlop_mul_simple(arg_t a, arg_t b)
{
	return snlop_mul(a, b, 0);
}



arg_t snlop_div_F(arg_t a, arg_t b, unsigned long flags)
{
	arg_t inv = snlop_inv_F(b);
	return snlop_mul_F(a, inv, flags);
}


arg_t snlop_axpbz(arg_t a, arg_t b, complex float sa, complex float sb)
{
	const struct iovec_s* iova = arg_get_iov(a);
	const struct iovec_s* iovb = arg_get_iov(b);

	int N = MAX(iova->N, iovb->N);
	long adims[N];
	long bdims[N];
	long mdims[N];

	md_singleton_dims(N, adims);
	md_singleton_dims(N, bdims);

	md_copy_dims(iova->N, adims, iova->dims);
	md_copy_dims(iovb->N, bdims, iovb->dims);

	md_max_dims(N, ~0UL, mdims, adims, bdims);

	const struct nlop_s* nlop = nlop_zaxpbz2_create(N, mdims, md_nontriv_dims(N, adims), sa, md_nontriv_dims(N, bdims), sb);

	nlop = nlop_reshape_in_F(nlop, 0, iova->N, iova->dims);
	nlop = nlop_reshape_in_F(nlop, 1, iovb->N, iovb->dims);

	return snlop_append_nlop_generic_F(2, (arg_t[2]){ a, b }, nlop, true);
}

arg_t snlop_axpbz_F(arg_t a, arg_t b, complex float sa, complex float sb)
{
	const struct iovec_s* iova = arg_get_iov(a);
	const struct iovec_s* iovb = arg_get_iov(b);

	int N = MAX(iova->N, iovb->N);
	long adims[N];
	long bdims[N];
	long mdims[N];

	md_singleton_dims(N, adims);
	md_singleton_dims(N, bdims);

	md_copy_dims(iova->N, adims, iova->dims);
	md_copy_dims(iovb->N, bdims, iovb->dims);

	md_max_dims(N, ~0UL, mdims, adims, bdims);

	const struct nlop_s* nlop = nlop_zaxpbz2_create(N, mdims, md_nontriv_dims(N, adims), sa, md_nontriv_dims(N, bdims), sb);

	nlop = nlop_reshape_in_F(nlop, 0, iova->N, iova->dims);
	nlop = nlop_reshape_in_F(nlop, 1, iovb->N, iovb->dims);

	return snlop_append_nlop_generic_F(2, (arg_t[2]){ a, b }, nlop, false);
}

arg_t snlop_add(arg_t a, arg_t b)
{
	return snlop_axpbz(a, b, 1., 1.);
}

arg_t snlop_add_F(arg_t a, arg_t b)
{
	return snlop_axpbz_F(a, b, 1., 1.);
}

arg_t snlop_sub(arg_t a, arg_t b)
{
	return snlop_axpbz(a, b, 1., -1.);
}

arg_t snlop_sub_F(arg_t a, arg_t b)
{
	return snlop_axpbz_F(a, b, 1., -1.);
}

arg_t snlop_dump(arg_t arg, const char* name, bool frw, bool der, bool adj)
{
	const struct iovec_s* iov = arg_get_iov(arg);

	auto dump = nlop_dump_create(iov->N, iov->dims, name, frw, der, adj);

	return snlop_append_nlop_F(arg, dump, true);
}