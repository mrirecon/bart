/* Copyright 2024-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>
#include <math.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/multiplace.h"
#include "num/flpmath.h"
#include "num/ops_p.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "iter/thresh.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/prox.h"

#include "motion/displacement.h"
#include "motion/interpolate.h"
#include "motion/pyramide.h"

#include "opticalflow.h"



static void zentral_differences(int D, const long dims[D], int d, unsigned long flags, complex float* out, const complex float* in)
{
	long idims[D];
	md_select_dims(D, ~MD_BIT(d), idims, dims);

	const struct linop_s* lop = linop_grad_zentral_create(D, idims, d, flags);

	linop_forward(lop, D, dims, out, D, idims, in);

	linop_free(lop);
}

/**
 * data consistency: I0(x) - I1(x + u(x))
 * I0(x) - static image
 * I1(x) - moved image
 * linearization: I0(x) - I1(x + u0(x)) - grad I1|u0(x) * (u(x) - u0(x))
 * 		  = rho0(x) - dimg_moved(x) * u(x)
 * with rho0(x) 	= I0(x) - I1(x + u0(x)) - grad I1|u0(x) * u0(x)
 * 	dimg_moved(x)	= grad I1|u0(x)
 */


/**
 * Proximal function for f(z) = lambda / p * || rho0 + sum_i (dimg_moved_i * z_i) ||_p^p, p=1,2
 * Only supports real valued input.
 */
struct prox_img_data {

	operator_data_t super;

	float lambda;

	int N;
	int d;
	const long* dims;

	struct multiplace_array_s* rho0;
	struct multiplace_array_s* dimg_moved;
};

static DEF_TYPEID(prox_img_data);

//dst = (1 + lambda (dimg * dimg^H))^(-1) (src + lambda dimg * rho0)
//    = [1 - (dimg * dimg^H) / (dimg^H * dimg)] (src + lambda dimg * rho0)
static void prox_img_l2_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(prox_img_data, _data);

	mu *= d->lambda;

	int N = d->N;
	const long* dims = d->dims;
	long img_dims[N];
	md_select_dims(N, ~MD_BIT(d->d), img_dims, dims);

	long strs[N];
	long img_strs[N];

	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);

	const complex float* rho0 = multiplace_read(d->rho0, src);
	const complex float* dimg_moved = multiplace_read(d->dimg_moved, src);

	complex float* tmp = md_alloc_sameplace(N, img_dims, CFL_SIZE, dst);
	complex float* tmp_img = md_alloc_sameplace(N, img_dims, CFL_SIZE, dst);

	md_copy(N, dims, dst, src, CFL_SIZE);

	md_ztenmul(N, dims, tmp, dims, dimg_moved, img_dims, rho0);
	md_zaxpy(N, dims, dst, mu, tmp); // dst = src + lambda dimg * rho0

	md_ztenmulc(N, img_dims, tmp_img, dims, dimg_moved, dims, dimg_moved);
	md_zreal(N, img_dims, tmp_img, tmp_img);
	md_zsqrt(N, img_dims, tmp_img, tmp_img);
	md_zdiv2(N, dims, strs, tmp, strs, dimg_moved, img_strs, tmp_img);

	md_ztenmulc(N, img_dims, tmp_img, dims, dst, dims, dst);
	md_zsmul(N, img_dims, tmp_img, tmp_img, -1.);
	md_zfmac2(N, dims, strs, dst, strs, tmp, img_strs, tmp_img);

	md_free(tmp);
	md_free(tmp_img);
}


static void prox_img_l1_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	auto d = CAST_DOWN(prox_img_data, _data);

	mu *= d->lambda;

	int N = d->N;
	const long* dims = d->dims;
	long img_dims[N];
	md_select_dims(N, ~MD_BIT(d->d), img_dims, dims);

	long strs[N];
	long img_strs[N];

	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);

	const complex float* rho0 = multiplace_read(d->rho0, src);
	const complex float* dimg_moved = multiplace_read(d->dimg_moved, src);

	complex float* rho = md_alloc_sameplace(N, img_dims, CFL_SIZE, dst);
	md_ztenmul(N, img_dims, rho, dims, dimg_moved, dims, src);
	md_zadd(N, img_dims, rho, rho0, rho);

	complex float* gnorm = md_alloc_sameplace(N, img_dims, CFL_SIZE, dst);
	md_ztenmulc(N, img_dims, gnorm, dims, dimg_moved, dims, dimg_moved);
	md_zsmul(N, img_dims, gnorm, gnorm, mu);

	md_copy(N, dims, dst, src, CFL_SIZE);

	complex float* case1 = md_alloc_sameplace(N, img_dims, CFL_SIZE, dst);
	complex float* case2 = md_alloc_sameplace(N, img_dims, CFL_SIZE, dst);
	complex float* case3 = md_alloc_sameplace(N, img_dims, CFL_SIZE, dst);

	md_zgreatequal(N, img_dims, case2, rho, gnorm);

	md_zsmul(N, img_dims, gnorm, gnorm, -1.);

	md_zgreatequal(N, img_dims, case1, gnorm, rho);

	md_zfill(N, img_dims, case3, 1.);
	md_zsub(N, img_dims, case3, case3, case1);
	md_zsub(N, img_dims, case3, case3, case2);

	md_zsub(N, img_dims, case1, case1, case2);
	md_zsmul(N, img_dims, case1, case1, mu);
	md_zfmac2(N, dims, strs, dst, strs, dimg_moved, img_strs, case1);

	md_zmul(N, img_dims, case3, case3, rho);
	md_zsmul(N, img_dims, case3, case3, -mu);
	md_zdiv(N, img_dims, case3, case3, gnorm);

	md_zfmac2(N, dims, strs, dst, strs, dimg_moved, img_strs, case3);

	md_zreal(N, dims, dst, dst);

	md_free(rho);
	md_free(gnorm);
	md_free(case1);
	md_free(case2);
	md_free(case3);
}

static void prox_img_del(const operator_data_t* _data)
{
	auto d = CAST_DOWN(prox_img_data, _data);

	multiplace_free(d->rho0);
	multiplace_free(d->dimg_moved);

	xfree(d->dims);

	xfree(d);
}

static const struct operator_p_s* prox_img_create(bool l1, float lambda, int d, int N, const long dims[N], const complex float* rho0, const complex float* dimg_moved)
{
	PTR_ALLOC(struct prox_img_data, data);
	SET_TYPEID(prox_img_data, data);

	data->N = N;
	data->d = d;
	data->dims = ARR_CLONE(long[N], dims);
	data->lambda = lambda;

	long img_dims[N];
	md_select_dims(N, ~MD_BIT(d), img_dims, dims);

	data->rho0 = multiplace_move(N, img_dims, CFL_SIZE, rho0);
	data->dimg_moved = multiplace_move(N, dims, CFL_SIZE, dimg_moved);

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), l1 ? prox_img_l1_apply : prox_img_l2_apply, prox_img_del);
}





void optical_flow(bool l1_reg, unsigned long reg_flags, float lambda, float maxnorm, bool l1_dc, int d, unsigned long flags, int N, const long dims[N], const complex float* img_static, const complex float* _img_moved, complex float* u)
{
	long img_dims[N];
	md_select_dims(N, ~MD_BIT(d), img_dims, dims);

	complex float* rho0 = md_alloc_sameplace(N, img_dims, CFL_SIZE, _img_moved);

	const struct linop_s* lop_motion = linop_interpolate_displacement_create(d, flags, 3, N, img_dims, dims, u, img_dims);
	linop_forward(lop_motion, N, img_dims, rho0, N, img_dims, _img_moved);
	linop_free(lop_motion);

	complex float* dimg_moved = md_alloc_sameplace(N, dims, CFL_SIZE, _img_moved);
	zentral_differences(N, dims, d, flags, dimg_moved, rho0);

	long strs[N];
	long img_strs[N];

	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, img_strs, img_dims, CFL_SIZE);

	md_zsmul(N, dims, dimg_moved, dimg_moved, -1.);
	md_zfmac2(N, dims, img_strs, rho0, strs, dimg_moved, strs, u);
	md_zsmul(N, dims, dimg_moved, dimg_moved, -1.);

	md_zsub(N, img_dims, rho0, rho0, img_static);

	const struct operator_p_s* prox_img = prox_img_create(l1_dc, 1., d, N, dims, rho0, dimg_moved);

	md_free(rho0);
	md_free(dimg_moved);

	const struct operator_p_s* prox_ops[3];
	const struct linop_s* trafos[3];

	trafos[0] = linop_identity_create(N, dims);
	prox_ops[0] = prox_img;

	trafos[1] = linop_grad_create(N, dims, N, reg_flags);

	if (l1_reg)
		prox_ops[1] = prox_thresh_create(N + 1,	linop_codomain(trafos[1])->dims, lambda, 0);
	else
		prox_ops[1] = prox_leastsquares_create(N + 1,linop_codomain(trafos[1])->dims, lambda, NULL);

	trafos[2] = linop_identity_create(N, dims);
	prox_ops[2] = prox_l2ball2_create(N, ~MD_BIT(d), dims, maxnorm, NULL);

	struct iter_chambolle_pock_conf iconf = iter_chambolle_pock_defaults;
	float scale = 4 * bitcount(reg_flags);

	if (0 < maxnorm)
		scale += 1.;

	iconf.sigma /= sqrtf(scale);
	iconf.tau /= sqrtf(scale);

	iter2_chambolle_pock(CAST_UP(&iconf), NULL, (0 < maxnorm ? 3 : 2), prox_ops, trafos, NULL, NULL, 2 * md_calc_size(N, dims), (float*)u, NULL, NULL);

	linop_free(trafos[0]);
	linop_free(trafos[1]);
	linop_free(trafos[2]);

	operator_p_free(prox_ops[0]);
	operator_p_free(prox_ops[1]);
	operator_p_free(prox_ops[2]);
}



void optical_flow_multiscale(bool l1_reg, unsigned long reg_flags, float lambda, float maxnorm, bool l1_dc,
			     int levels, float sigma[levels], float factors[levels], int nwarps[levels],
			     int d, unsigned long flags, int N, const long _dims[N], const complex float* _img_static, const complex float* _img_moved, complex float* _u)
{
	assert(_dims[d] == bitcount(flags));

	long tdims[N];
	md_select_dims(N, ~MD_BIT(d), tdims, _dims);

	long dims[levels][N];
	complex float* img_static[levels];
	complex float* img_moved[levels];

	gaussian_pyramide(levels, factors, sigma, 3, N, flags, tdims, _img_moved, dims, img_moved);
	gaussian_pyramide(levels, factors, sigma, 3, N, flags, tdims, _img_static, dims, img_static);

	long udims[N];
	md_copy_dims(N, udims, dims[levels - 1]);
	udims[d] = _dims[d];

	complex float* u = md_alloc_sameplace(N, udims, CFL_SIZE, _u);
	md_clear(N, udims, u, CFL_SIZE);

	long fdims[N];
	md_select_dims(N, MD_BIT(d), fdims, _dims);

	complex float factors_cpu[fdims[d]];
	complex float* factors_sp = md_alloc_sameplace(N, fdims, CFL_SIZE, u);

	for (int i = levels - 1; i >= 0; i--) {

		for (int j = 0; j < nwarps[i]; j++)
			optical_flow(l1_reg, reg_flags, lambda, maxnorm * factors[i], l1_dc, d, flags, N, udims, img_static[i], img_moved[i], u);

		md_free(img_moved[i]);
		md_free(img_static[i]);

		if (0 == i)
			break;

		long nudims[N];
		md_copy_dims(N, nudims, dims[i - 1]);
		nudims[d] = _dims[d];

		complex float* u2 = md_alloc_sameplace(N, nudims, CFL_SIZE, u);

		md_resample(flags, 3, N, nudims, u2, udims, u);

		md_free(u);

		u = u2;
		md_copy_dims(N, udims, nudims);

		for (int j = 0, jp = 0; j < N; j++)
			if (MD_IS_SET(flags, j))
				factors_cpu[jp++] = (float)dims[i -1][j] / (float)dims[i][j];

		md_copy(N, fdims, factors_sp, factors_cpu, CFL_SIZE);
		md_zmul2(N, udims, MD_STRIDES(N, udims, CFL_SIZE), u, MD_STRIDES(N, udims, CFL_SIZE), u, MD_STRIDES(N, fdims, CFL_SIZE), factors_sp);
	}


	md_free(factors_sp);

	md_copy(N, udims, _u, u, CFL_SIZE);
	md_free(u);
}




