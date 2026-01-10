/* Copyright 2024-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
 * Publications:
 *
 * E. Parzen. On the estimation of a probability density
 * function and the mode. Annals of Mathematical Statistics,
 * 33(3), 1065-1076, 1962.
 *
 * Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
 * PET-CT image registration in the chest using free-form deformations.
 * IEEE TMI, 22(1), 120-8, 2003.
 */

#include <math.h>
#include <complex.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "nlops/nlop.h"

#include "mi_metric.h"


static float bin_normalize(float x, float mval, float delta);
static inline int bin_index(float normalized, int nbins, int padding);
static inline float cubic_spline(float x);
static inline float cubic_spline_derivative(float x);

// Mutual Information metric with Parzan joint histogram
struct mi_metric_s {

	nlop_data_t super;

	int nbins;
	int padding;

	float smin;
	float smax;
	float sdelta;

	float mmin;
	float mmax;
	float mdelta;

	double* joint;
	double* smarginal;
	double* mmarginal;

	long tot;
	float* img_static;
	float* img_moving;
	float* msk_static;
	float* msk_moving;

	long valid_points;
};

DEF_TYPEID(mi_metric_s);




static void mim_del(const nlop_data_t* _data)
{
	const auto mim = CAST_DOWN(mi_metric_s, _data);

	md_free(mim->img_static);
	md_free(mim->img_moving);
	md_free(mim->msk_static);
	md_free(mim->msk_moving);

	xfree(mim->joint);
	xfree(mim->smarginal);
	xfree(mim->mmarginal);

	xfree(mim);
}


static void mim_forward(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto mim = CAST_DOWN(mi_metric_s, _data);

	complex float* dst = args[0];
	const complex float* _img_moving = args[1];
	const complex float* _img_static = args[2];

	md_real(1, MD_DIMS(mim->tot), mim->img_moving, _img_moving);
	md_real(1, MD_DIMS(mim->tot), mim->img_static, _img_static);
	
	if (5 == N) {

		md_real(1, MD_DIMS(mim->tot), mim->msk_moving, args[3]);
		md_real(1, MD_DIMS(mim->tot), mim->msk_static, args[4]);
	}

	double (*joint)[mim->nbins][mim->nbins] = (void*)mim->joint;

	for (int i = 0; i < mim->nbins; i++)
		for (int j = 0; j < mim->nbins; j++)
			(*joint)[i][j] = 0;

	for (int i = 0; i < mim->nbins; i++)
		mim->smarginal[i] = 0;


	double total_sum = 0;

	mim->valid_points = 0;

	for (int i = 0; i < mim->tot; i++) {

		if (mim->msk_static && (0 == mim->msk_static[i]))
			continue;

		if (mim->msk_moving && (0 == mim->msk_moving[i]))
			continue;

		mim->valid_points += 1;

		double rn = bin_normalize(mim->img_static[i], mim->smin, mim->sdelta);
		long r = bin_index(rn, mim->nbins, mim->padding);
		double cn = bin_normalize(mim->img_moving[i], mim->mmin, mim->mdelta);
		long c = bin_index(cn, mim->nbins, mim->padding);
		double spline_arg = (c - 2) - cn;

		mim->smarginal[r] += 1;

		for (int offset = -2; offset < 3; offset++) {

			float val = cubic_spline(spline_arg);
			(*joint)[r][c + offset] += val;
			total_sum += val;
			spline_arg += 1.0;
		}
	}


	if (0 < total_sum) {

		for (int i = 0; i < mim->nbins; i++)
			for (int j = 0; j < mim->nbins; j++)
				(*joint)[i][j] /= total_sum;

		for (int i = 0; i < mim->nbins; i++)
			mim->smarginal[i] /= mim->valid_points;

		for (int j = 0; j < mim->nbins; j++) {

			mim->mmarginal[j] = 0;
			for (int i = 0; i < mim->nbins; i++)
				mim->mmarginal[j] += (*joint)[i][j];
		}
	}

	double epsilon = 2.2204460492503131e-016;
	double metric_value = 0;

	for (int i = 0; i < mim->nbins; i++)
		for (int j = 0; j < mim->nbins; j++) {

			if (((*joint)[i][j] < epsilon) || (mim->mmarginal[j] < epsilon))
				continue;

			double factor = log((*joint)[i][j] / mim->mmarginal[j]);

			if (mim->smarginal[i] > epsilon)
				metric_value += (*joint)[i][j] * (factor - log(mim->smarginal[i]));
		}

	dst[0] = -metric_value;
}


static void mim_gradient(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{

	const auto mim = CAST_DOWN(mi_metric_s, _data);

	double epsilon = 2.2204460492503131e-016;

	double (*joint)[mim->nbins][mim->nbins] = (void*)mim->joint;
	double djoint[mim->nbins][mim->nbins];

	for (int i = 0; i < mim->nbins; i++)
		for (int j = 0; j < mim->nbins; j++)
			if (((*joint)[i][j] < epsilon) || (mim->mmarginal[j] < epsilon))
				djoint[i][j] = 0;
			else
				djoint[i][j] = log((*joint)[i][j] / mim->mmarginal[j]);

	float norm_factor = (mim->valid_points > 0) ? 1. / (mim->valid_points * mim->mdelta) : 0;


#pragma omp parallel for
	for (long i = 0; i < mim->tot; i++) {

		dst[i] = 0.;

		if (mim->msk_static && (0 == mim->msk_static[i]))
			continue;

		if (mim->msk_moving && (0 == mim->msk_moving[i]))
			continue;

		float rn = bin_normalize(mim->img_static[i], mim->smin, mim->sdelta);
		int r = bin_index(rn, mim->nbins, mim->padding);
		float cn = bin_normalize(mim->img_moving[i], mim->mmin, mim->mdelta);
		int c = bin_index(cn, mim->nbins, mim->padding);
		float spline_arg = (c - 2) - cn;

		for (int offset = -2; offset < 3; offset++) {

			float val = cubic_spline_derivative(spline_arg);

			dst[i] -= djoint[r][c + offset] * val;

			spline_arg += 1.0;
		}

		dst[i] *= -norm_factor * crealf(src[0]);
	}
}


static float bin_normalize(float x, float mval, float delta)
{
	if (0 == delta)
		return 0;
	else
		return x / delta - mval;
}


static inline int bin_index(float normalized, int nbins, int padding)
{
	long bin_id = normalized;

	if (bin_id < padding)
		return padding;

	if (bin_id > nbins - 1 - padding)
		return nbins - 1 - padding;

	return bin_id;
}


static inline float cubic_spline(float x)
{
	float absx = fabs(x);
	float sqrx = x * x;

	if (1.0 > absx)
		return (4.0 - 6.0 * sqrx + 3.0 * sqrx * absx) / 6.0;

	if (2.0 > absx)
		return (8.0 - 12 * absx + 6.0 * sqrx - sqrx * absx) / 6.0;

	return 0.0;
}


static inline float cubic_spline_derivative(float x)
{
       float absx = fabs(x);

	if (1. > absx) {

		if (0. <= x)
	    		return -2.0 * x + 1.5 * x * x;
		else
	    		return -2.0 * x - 1.5 * x * x;
	}

	if (2. > absx) {

		if (0. <= x)
			return -2.0 + 2.0 * x - 0.5 * x * x;
		else
	    		return 2.0 + 2.0 * x + 0.5 * x * x;
	}

	return 0.0;
}


struct nlop_s* nlop_mi_metric_create(int N, const long dims[N], int nbins, float smin, float smax, float mmin, float mmax, bool mask)
{
	PTR_ALLOC(struct mi_metric_s, mim);
	SET_TYPEID(mi_metric_s, mim);

	mim->tot = md_calc_size(N, dims);

	mim->nbins = nbins;
	mim->padding = 2;

	mim->img_moving = NULL;
	mim->img_static = NULL;

	mim->smin = smin;
	mim->smax = smax;
	mim->mmin = mmin;
	mim->mmax = mmax;

	mim->sdelta = (mim->smax - mim->smin) / (mim->nbins - 2 * mim->padding);
	mim->mdelta = (mim->mmax - mim->mmin) / (mim->nbins - 2 * mim->padding);

	mim->smin = mim->smin / mim->sdelta - mim->padding;
	mim->mmin = mim->mmin / mim->mdelta - mim->padding;

	mim->joint = &(*TYPE_ALLOC(double[nbins][nbins])[0][0]);
	mim->smarginal = *TYPE_ALLOC(double[nbins]);
	mim->mmarginal = *TYPE_ALLOC(double[nbins]);

	mim->img_moving = md_alloc(1, MD_DIMS(mim->tot), FL_SIZE);
	mim->img_static = md_alloc(1, MD_DIMS(mim->tot), FL_SIZE);
	mim->msk_moving = mask ? md_alloc(1, MD_DIMS(mim->tot), FL_SIZE) : NULL;
	mim->msk_static = mask ? md_alloc(1, MD_DIMS(mim->tot), FL_SIZE) : NULL;

	long nl_odims[1][1] = { { 1 } };

	long nl_idims[4][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);
	md_copy_dims(N, nl_idims[2], dims);
	md_copy_dims(N, nl_idims[3], dims);

	struct nlop_s* ret = nlop_generic_create(1, 1, nl_odims, mask ? 4 : 2, N, nl_idims, CAST_UP(PTR_PASS(mim)),
		mim_forward, NULL, mask ? (nlop_der_fun_t[2][1]){ { mim_gradient }, { NULL } } : (nlop_der_fun_t[4][1]){ { mim_gradient }, { NULL }, { NULL }, { NULL } }, NULL, NULL, mim_del);

	if (mask) {

		ret = (struct nlop_s*)nlop_no_der_F(ret, 0, 2);
		ret = (struct nlop_s*)nlop_no_der_F(ret, 0, 3);
	}

	return ret;
}









