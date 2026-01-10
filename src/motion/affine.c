/* Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
 *
 * Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
 * PET-CT image registration in the chest using free-form deformations.
 * IEEE TMI 2003;22:120-8.
 */

#include <assert.h>
#include <math.h>
#include <float.h>
#include <complex.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/lapack.h"
#include "num/multind.h"
#include "num/iovec.h"
#include "num/loop.h"

#include "iter/italgos.h"
#include "iter/iter3.h"
#include "iter/iter4.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/stack.h"
#include "nlops/const.h"
#include "nlops/tenmul.h"
#include "nlops/cast.h"
#include "nlops/mi_metric.h"

#include "motion/interpolate.h"

#include "affine.h"

/**
 * Affine transformations are stored as complex 3x4 matrices using md_*
 * conventions (i.e. Fortran order).
 *
 *     (\ | / | | )
 * A = (- R - | a )
 *     (/ | \ | | )
 *
 * Applying an affine transform means sampling the interpolated image at 
 * coordinates A * pos where pos is the position in the output image.
 * Affine transformations are defined in units of FoV and the origin is defined
 * by the center of the FFT.
 *
 **/


//C = AB
const struct nlop_s* nlop_affine_chain_FF(const struct nlop_s* A, const struct nlop_s* B)
{
	complex float add[4] = { 0, 0., 0., 1. };

	A = nlop_reshape_out_F(A, 0, 3, MD_DIMS(3, 4, 1));
	
	B = nlop_combine_FF(B, nlop_const_create(2, MD_DIMS(1, 4), true, add));
	B = nlop_stack_outputs_F(B, 0, 1, 0);
	B = nlop_reshape_out_F(B, 0, 3, MD_DIMS(1, 4, 4));

	const struct nlop_s* C = nlop_tenmul_create(3, MD_DIMS(3, 1, 4), MD_DIMS(3, 4, 1), MD_DIMS(1, 4, 4));

	C = nlop_chain2_FF(A, 0, C, 0);
	C = nlop_chain2_FF(B, 0, C, 0);

	if (1 < nlop_get_nr_in_args(C))
		C = nlop_stack_inputs_F(C, 0, 1, 0);

	C = nlop_reshape_out_F(C, 0, 2, MD_DIMS(3, 4));

	return C;
}


const struct nlop_s* nlop_affine_prepend_FF(const struct nlop_s* A, complex float* B)
{
	return nlop_affine_chain_FF(A, nlop_const_create(2, MD_DIMS(3, 4), true, B));
}

const struct nlop_s* nlop_affine_append_FF(complex float* A, const struct nlop_s* B)
{
	return nlop_affine_chain_FF(nlop_const_create(2, MD_DIMS(3, 4), true, A), B);
}


static void affine_set(int i, int j, complex float* dst, float val)
{
	dst[i + 3 * j] = val;
}

static float affine_get(int i, int j, const complex float* src)
{
	return crealf(src[i + 3 * j]);
}

void affine_debug(int dl, const complex float* A)
{
	debug_printf(dl, "Affine Transform Matrix:\n");

	for (int i = 0; i < 3; i++) {

		for (int j = 0; j < 4; j++)
			debug_printf(dl, "  %+.2e", affine_get(i, j, A));

		debug_printf(dl, "\n");
	}
}

static void affine_chain_complex(complex float* C, const complex float* A, const complex float* B)
{
	for (int i = 0; i < 3; i++) {

		for (int j = 0; j < 4; j++) {

			affine_set(i, j, C, 0.);

			for (int k = 0; k < 3; k++)
				affine_set(i, j, C, affine_get(i, j, C) +  affine_get(i, k, A) *  affine_get(k, j, B) );
		}

		affine_set(i, 3, C, affine_get(i, 3, C) +  affine_get(i, 3, A));
	}
}

void affine_init_id(complex float* dst)
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			affine_set(i, j, dst,  i == j ? 1 : 0);
}

static void affine_init_zero(complex float* dst)
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			affine_set(i, j, dst, 0);
}

static void affine_grid2world(const long dims[3], complex float* aff)
{
	affine_init_id(aff);

	for (int i = 0; i < 3; i++) {

		affine_set(i, i, aff, 1. / dims[i]);
		affine_set(i, 3, aff, -((float)(dims[i] / 2)) / dims[i]);
	}
}

static void affine_world2grid(const long dims[3], complex float* aff)
{
	affine_init_id(aff);

	for (int i = 0; i < 3; i++) {

		affine_set(i, i, aff, dims[i]);
		affine_set(i, 3, aff, ((float)(dims[i] / 2)));
	}
}

static const struct nlop_s* nlop_affine_grid2world(const long dims[3])
{
	complex float aff[12];
	affine_init_id(aff);

	for (int i = 0; i < 3; i++) {

		affine_set(i, i, aff, 1. / dims[i]);
		affine_set(i, 3, aff, -((float)(dims[i] / 2)) / dims[i]);
	}

	return nlop_const_create(2, MD_DIMS(3, 4), true, aff);
}

static const struct nlop_s* nlop_affine_world2grid(const long dims[3])
{
	complex float aff[12];
	affine_init_id(aff);

	for (int i = 0; i < 3; i++) {

		affine_set(i, i, aff, dims[i]);
		affine_set(i, 3, aff, ((float)(dims[i] / 2)));
	}
	
	return nlop_const_create(2, MD_DIMS(3, 4), true, aff);
}

const struct nlop_s* nlop_affine_to_grid_F(const struct nlop_s* affine, const long sdims[3], const long mdims[3])
{
	affine = nlop_affine_chain_FF(affine, nlop_affine_grid2world(sdims));
	affine = nlop_affine_chain_FF(nlop_affine_world2grid(mdims), affine);
	return affine;
}


struct affine_s {

	nlop_data_t super;

	int idx; // for rotation

	int Npars;
	float* pars;
};

DEF_TYPEID(affine_s);

static void affine_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(affine_s, _data);

	xfree(data->pars);
	xfree(data);
}

static const struct nlop_s* nlop_affine_create(int Npars, int idx, nlop_fun_t fun, nlop_der_fun_t der, nlop_der_fun_t adj)
{
	PTR_ALLOC(struct affine_s, data);
	SET_TYPEID(affine_s, data);

	data->idx = idx;
	data->Npars = Npars;
	data->pars = *TYPE_ALLOC(float[Npars]);

	return nlop_cpu_wrapper_F(nlop_create(2, MD_DIMS(3, 4), 1, MD_DIMS(Npars), CAST_UP(PTR_PASS(data)), fun, der, adj, NULL, NULL, affine_del));
}




static void affine_translation_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);
	
	for (int i = 0; i < d->Npars; i++)
		d->pars[i] = crealf(src[i]);

	affine_init_id(dst);

	for (int i = 0; i < d->Npars; i++)
		affine_set(i, 3, dst, d->pars[i]);
}

static void affine_translation_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);
	
	affine_init_zero(dst);
	for (int i = 0; i < d->Npars; i++)
		affine_set(i, 3, dst, crealf(src[i]));
}

static void affine_translation_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);
	
	for (int i = 0; i < d->Npars; i++)
		dst[i] = affine_get(i, 3, src);
}

const struct nlop_s* nlop_affine_translation_2D(void)
{
	return nlop_affine_create(2, 0, affine_translation_fun, affine_translation_der, affine_translation_adj);	
}

const struct nlop_s* nlop_affine_translation_3D(void)
{
	return nlop_affine_create(3, 0, affine_translation_fun, affine_translation_der, affine_translation_adj);	
}



static void affine_rot_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);
	
	for (int i = 0; i < d->Npars; i++)
		d->pars[i] = crealf(src[i]);

	int a1 = (d->idx + 1) % 3;
	int a2 = (d->idx + 2) % 3;

	float s = sin(d->pars[0]);
	float c = cos(d->pars[0]);

	affine_init_id(dst);

	affine_set(a1, a1, dst, c);
	affine_set(a2, a2, dst, c);
	affine_set(a1, a2, dst, -s);
	affine_set(a2, a1, dst, s);
}

static void affine_rot_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);
	
	int a1 = (d->idx + 1) % 3;
	int a2 = (d->idx + 2) % 3;

	float s = cos(d->pars[0]) * crealf(src[0]);
	float c = -sin(d->pars[0]) * crealf(src[0]);

	affine_init_zero(dst);

	affine_set(a1, a1, dst, c);
	affine_set(a2, a2, dst, c);
	affine_set(a1, a2, dst, -s);
	affine_set(a2, a1, dst, s);
}

static void affine_rot_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);

	int a1 = (d->idx + 1) % 3;
	int a2 = (d->idx + 2) % 3;

	float s = cosf(d->pars[0]);
	float c = -sinf(d->pars[0]);

	dst[0] = 0;
	dst[0] += c * affine_get(a1, a1, src);
	dst[0] += c * affine_get(a2, a2, src);
	dst[0] -= s * affine_get(a1, a2, src);
	dst[0] += s * affine_get(a2, a1, src);
}


const struct nlop_s* nlop_affine_rotation_2D(void)
{
	return nlop_affine_create(1, 2, affine_rot_fun, affine_rot_der, affine_rot_adj);	
}

const struct nlop_s* nlop_affine_rotation_3D(void)
{
	const struct nlop_s* ret = nlop_affine_create(1, 2, affine_rot_fun, affine_rot_der, affine_rot_adj);
	ret = nlop_affine_chain_FF(nlop_affine_create(1, 0, affine_rot_fun, affine_rot_der, affine_rot_adj), ret);
	ret = nlop_affine_chain_FF(nlop_affine_create(1, 2, affine_rot_fun, affine_rot_der, affine_rot_adj), ret);

	return ret;
}


const struct nlop_s* nlop_affine_rigid_2D(void)
{
	return nlop_affine_chain_FF(nlop_affine_translation_2D(), nlop_affine_rotation_2D());
}

const struct nlop_s* nlop_affine_rigid_3D(void)
{
	return nlop_affine_chain_FF(nlop_affine_translation_3D(), nlop_affine_rotation_3D());
}




static void affine_affine_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);
	
	int dims = (12 == d->Npars) ? 3 : 2;
	affine_init_id(dst);

	for (int i = 0; i < dims; i++)
		for (int j = 0; j < dims; j++)
			affine_set(i, j, dst, (i == j) ? src[i + dims * j] + 1 : src[i + dims * j]); // plus one to have identity for pars=0

	for (int i = 0; i < dims; i++)
		affine_set(i, 3, dst, src[i + dims * dims]);
}

static void affine_affine_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);
	
	int dims = (12 == d->Npars) ? 3 : 2;
	affine_init_zero(dst);

	for (int i = 0; i < dims; i++)
		for (int j = 0; j < dims; j++)
			affine_set(i, j, dst, src[i + dims * j]);

	for (int i = 0; i < dims; i++)
		affine_set(i, 3, dst, src[i + dims * dims]);
}

static void affine_affine_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(affine_s, _data);
	
	int dims = (12 == d->Npars) ? 3 : 2;

	for (int i = 0; i < dims; i++)
		for (int j = 0; j < dims; j++)
			dst[i + dims * j] = affine_get(i, j, src);

	for (int i = 0; i < dims; i++)
		dst[i + dims * dims] = affine_get(i, 3, src);
}

const struct nlop_s* nlop_affine_2D(void)
{
	return nlop_affine_create(6, 0, affine_affine_fun, affine_affine_der, affine_affine_adj);	
}

const struct nlop_s* nlop_affine_3D(void)
{
	return nlop_affine_create(12, 0, affine_affine_fun, affine_affine_der, affine_affine_adj);
}



const struct nlop_s* nlop_affine_compute_pos(int dim, int N, const long sdims[N], const long mdims[N], const struct nlop_s* affine)
{
	long dims[N + 1];
	md_copy_dims(N, dims, sdims);
	dims[N] = 4;

	NESTED(complex float, pos_kernel, (const long pos[]))
	{
		complex float ret = (3 == pos[dim + 1]) ? 1 : pos[pos[dim + 1]];

		return ret;
	};

	complex float* pos = md_alloc(N + 1, dims, CFL_SIZE);
	md_parallel_zsample(N + 1, dims, pos, pos_kernel);

	long pdims[N + 1];
	md_copy_dims(N, pdims, sdims);
	pdims[dim] = 3;
	pdims[N] = 1;

	long adims[N + 1];
	md_singleton_dims(N, adims);
	adims[dim] = 3;
	adims[N] = 4;

	affine = nlop_affine_chain_FF(affine, nlop_affine_grid2world(sdims));
	affine = nlop_affine_chain_FF(nlop_affine_world2grid(mdims), affine);

	auto lop = linop_fmac_dims_create(N + 1, pdims, adims, dims, pos);
	md_free(pos);

	lop = linop_reshape_in_F(lop, 2, MD_DIMS(3, 4));
	lop = linop_reshape_out_F(lop, N, pdims);

	return nlop_append_FF(affine, 0, nlop_from_linop_F(lop));
}

void affine_interpolate(int ord, const complex float* affine, const long _odims[3], complex float* dst, const long _idims[3], const complex float* src)
{
	long odims[4];
	long idims[4];
	long cdims[4];

	md_copy_dims(3, odims, _odims);
	md_copy_dims(3, idims, _idims);
	md_copy_dims(3, cdims, _odims);

	odims[3] = 1;
	idims[3] = 1;
	cdims[3] = 3;

	complex float g2w[12];
	complex float w2g[12];

	complex float tmp[12];
	complex float affine_grid[12];
	complex float* affine_grid_p = affine_grid;

	affine_grid2world(odims, g2w);
	affine_world2grid(idims, w2g);

	affine_chain_complex(tmp, affine, g2w);
	affine_chain_complex(affine_grid, w2g, tmp);

	NESTED(complex float, pos_kernel, (const long pos[]))
	{
		complex float ret = affine_get(pos[3], 3, affine_grid_p);

		for (int i = 0; i < 3; i++)
			ret += affine_get(pos[3], i, affine_grid_p) * pos[i];;

		return ret;
	};

	complex float* pos = md_alloc(4, cdims, CFL_SIZE);
	md_parallel_zsample(4, cdims, pos, pos_kernel);

	md_interpolate(3, 7, ord, 4, odims, dst, cdims, pos, idims, src);

	md_free(pos);
}




static const struct nlop_s* nlop_image_transform_affine_create(int ord, long _sdims[3], long _mdims[3], const struct nlop_s* trafo)
{

	long sdims[4];
	long mdims[4];
	long cdims[4];

	md_copy_dims(3, sdims, _sdims);
	md_copy_dims(3, mdims, _mdims);
	md_copy_dims(3, cdims, _sdims);

	sdims[3] = 1;
	mdims[3] = 1;
	cdims[3] = 3;

	const struct nlop_s* nlop  = nlop_clone(trafo);
	nlop = nlop_affine_compute_pos(3, 4, sdims, mdims, nlop);


	auto intp = nlop_interpolate_create(3, 7ul, ord, (1 == ord), 4, sdims, cdims, mdims);
	intp = nlop_reshape_in_F(intp, 0, 3, mdims);
	intp = nlop_reshape_out_F(intp, 0, 3, sdims);

	return nlop_prepend_FF(nlop, intp, 1);
}


static const struct nlop_s* affine_reg_nlop_create(
	      long sdims[3], const complex float* img_static, const complex float* msk_static,
	      long mdims[3], const complex float* img_moving, const complex float* msk_moving,
	      const struct nlop_s* trafo, bool gpu, bool cubic)
{
	float smin =  FLT_MAX;
	float smax = -FLT_MIN;
	float mmin =  FLT_MAX;
	float mmax = -FLT_MIN;

	long stot = sdims[0] * sdims[1] * sdims[2];
	long mtot = mdims[0] * mdims[1] * mdims[2];

	for (long i = 0; i < stot; i++) {

		if (NULL != msk_static)
			if (0. == msk_static[i])
				continue;

		smin = MIN(smin, cabsf(img_static[i]));
		smax = MAX(smax, cabsf(img_static[i]));
	}

	for (long i = 0; i < mtot; i++) {

		if (NULL != msk_moving)
			if (0. == msk_moving[i])
				continue;

		mmin = MIN(mmin, cabsf(img_moving[i]));
		mmax = MAX(mmax, cabsf(img_moving[i]));
	}

	auto nlop_mim = nlop_mi_metric_create(3, sdims, 32, smin, smax, mmin, mmax, NULL != msk_static);
	nlop_mim = nlop_set_input_const_F(nlop_mim, 1, 3, sdims, true, img_static);

	auto nlop_it = nlop_image_transform_affine_create(cubic ? 3 : 1, sdims, mdims, trafo);
	nlop_it = nlop_set_input_const_F(nlop_it, 0, 3, mdims, false, img_moving);

	if (gpu)
		nlop_it = nlop_gpu_wrapper_F(nlop_it);

	nlop_mim = nlop_prepend_FF(nlop_it, nlop_mim, 0);

	if (NULL != msk_static) {

		auto nlop_itmm = nlop_image_transform_affine_create(0, sdims, mdims, trafo);
		nlop_itmm = nlop_set_input_const_F(nlop_itmm, 0, 3, mdims, false, msk_moving);

		if (gpu)
			nlop_itmm = nlop_gpu_wrapper_F(nlop_itmm);

		nlop_mim = nlop_prepend_FF(nlop_itmm, nlop_mim, 1);
		nlop_mim = nlop_dup_F(nlop_mim, 0, 1);

		auto nlop_itms = nlop_image_transform_affine_create(0, sdims, sdims, trafo);
		nlop_itms = nlop_set_input_const_F(nlop_itms, 0, 3, sdims, false, msk_static);

		if (gpu)
			nlop_itms = nlop_gpu_wrapper_F(nlop_itms);

		nlop_mim = nlop_prepend_FF(nlop_itms, nlop_mim, 1);
		nlop_mim = nlop_dup_F(nlop_mim, 0, 1);
	}

	return nlop_mim;
}


static void gaussian_filter_3D(float sigma, const long dims[3], complex float* dst, const complex float* src)
{
	const struct linop_s* lop_conv = linop_identity_create(3, dims);

	for (int i = 0; i < 3; i++) {

		long fsize = MIN(8. * sigma + 1, dims[i]);

		complex float filter[fsize];

		float tot = 0;

		for (int i = 0; i < fsize; i++) {

			float x = (i - (fsize / 2)) / sigma;
			filter[i] = expf(-0.5 * x * x);
			tot += expf(-0.5 * x * x);
		}

		for (int i = 0; i < fsize; i++)
			filter[i] /= tot;

		long fdims[3] = { 1, 1, 1 };
		fdims[i] = fsize;

		lop_conv = linop_chain_FF(linop_conv_create(3, MD_BIT(i), CONV_TRUNCATED, CONV_SYMMETRIC, dims, dims, fdims, filter), lop_conv);
	}

	linop_forward_unchecked(lop_conv, dst, src);

	linop_free(lop_conv);
}





void affine_reg(bool gpu, bool cubic, complex float* affine, const struct nlop_s* _trafo, long sdims[3], const complex float* img_static, const complex float* msk_static, long mdims[3], const complex float* img_moving, const complex float* msk_moving, int N, float sigma[N], float factor[N])
{
	int npars = nlop_domain(_trafo)->dims[0];

	if ((NULL == msk_static) != (NULL == msk_moving))
		error("Need both masks or none.\n");

	for (int i = N - 1; i >= 0; i--) {

		auto trafo = nlop_affine_prepend_FF(nlop_clone(_trafo), affine);

		const complex float* wmsk_static = msk_static;
		const complex float* wimg_static = img_static;
		const complex float* wimg_moving = img_moving;

		long cdims[3];
		md_copy_dims(3, cdims, sdims);

		if (0. != sigma[i]) {

			complex float* simg_static = md_alloc(3, sdims, CFL_SIZE);
			gaussian_filter_3D(sigma[i], sdims, simg_static, img_static);

			for (int j = 0; j < 3; j++)
				cdims[j] = MAX(1, sdims[j] * factor[i]);

			complex float id[12];
			affine_init_id(id);

			complex float* cimg_static = md_alloc(3, sdims, CFL_SIZE);
			affine_interpolate(1, id, cdims, cimg_static, sdims, simg_static);

			md_free(simg_static);
			wimg_static = cimg_static;

			if (NULL != msk_static) {

				complex float* cmsk_static = md_alloc(3, sdims, CFL_SIZE);
				affine_interpolate(0, id, cdims, cmsk_static, sdims, msk_static);
				wmsk_static = cmsk_static;
			}

			complex float* simg_moving = md_alloc(3, mdims, CFL_SIZE);
			gaussian_filter_3D(sigma[i], sdims, simg_moving, img_moving);

			wimg_moving = simg_moving;
		}

		const struct nlop_s* nlop = affine_reg_nlop_create(cdims, wimg_static, wmsk_static, mdims, wimg_moving, msk_moving, trafo, gpu, cubic);

		struct iter3_lbfgs_conf conf = iter3_lbfgs_defaults;
		
		complex float pars[npars];
		md_clear(1, MD_DIMS(npars), pars, CFL_SIZE);

		iter4_lbfgs(CAST_UP(&conf), (struct nlop_s*)nlop, 2 * npars, (float*)pars, NULL, 2, NULL, NULL, (struct iter_op_s){ NULL, NULL });

		nlop_free(nlop);

		if (0. != sigma[i]) {

			md_free(wimg_moving);
			md_free(wimg_static);

			if (NULL != msk_static)
				md_free(wmsk_static);
		}

		nlop_apply(trafo, 2, MD_DIMS(3, 4), affine, 1, MD_DIMS(npars), pars);

		nlop_free(trafo);
	}
}

