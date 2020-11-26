/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/blas.h"
#include "num/iovec.h"
#include <math.h>

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "nlops/nlop.h"


#include "rbf.h"

struct rbf_s {

	INTERFACE(nlop_data_t);

	int N;				// N = 3
	const struct iovec_s* zdom;	// (Nf, Nb, 1 ) index convention: (i, k, j)
	const struct iovec_s* wdom;	// (Nf, 1,  Nw)

	bool use_imag;

	int idx_w; // 2;

	float Imax;
	float Imin;
	float sigma;

	float* z;
	float* dz;
};

DEF_TYPEID(rbf_s);

static void rbf_init(struct rbf_s* data, const complex float* ref)
{
	if (nlop_der_requested(CAST_UP(data), 0, 0)) {

		if (NULL == data->z)
			data->dz = md_alloc_sameplace(data->N, data->zdom->dims, data->zdom->size, ref);
		md_clear(data->N, data->zdom->dims, data->dz, data->zdom->size);

	} else {

		md_free(data->dz);
		data->dz = NULL;
	}

	if (nlop_der_requested(CAST_UP(data), 1, 0)) {

		if (NULL == data->z)
			data->z = md_alloc_sameplace(data->N, data->zdom->dims, data->zdom->size, ref);
		md_clear(data->N, data->zdom->dims, data->z, data->zdom->size);

	} else {

		md_free(data->z);
		data->z = NULL;
	}
}

static void rbf_clear_der(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(rbf_s, _data);

	md_free(data->dz);
	data->dz = NULL;

	md_free(data->z);
	data->z = NULL;
}



static void rbf_fun(const nlop_data_t* _data, int N_args, complex float* args[N_args])
{
	//dst_ik = sum_j w_ij * exp[-(z_ik-mu_j)^2/(s*sigma^2)]
	//data->dz_ik = sum_j (mu_j - z_ik)/sigma^2 * w_ij * exp[-(z_ik-mu_j)^2/(s*sigma^2)]


	assert(3 == N_args);
	const auto data = CAST_DOWN(rbf_s, _data);

	complex float* zdst = args[0];
	const complex float* zsrc = args[1];
	const complex float* wsrc = args[2];

	bool der1 = nlop_der_requested(_data, 0, 0);
	bool der2 = nlop_der_requested(_data, 1, 0);

	rbf_init(data, zdst);

	float* der_z = data->z;
	float* der_dz = data->dz;

	int N = data->N;
	const long* zdims = data->zdom->dims;
	const long* wdims = data->wdom->dims;
	const long* zstrs = data->zdom->strs;
	const long* wstrs = data->wdom->strs;

	float* tmp_w = md_alloc_sameplace(N, wdims, FL_SIZE, zdst);
	float* tmp_z = md_alloc_sameplace(N, zdims, FL_SIZE, zdst);

	long Nw = wdims[data->idx_w];
	float mumin = data->Imin;
	float dmu = (data->Imax - data->Imin)/((float)Nw - 1.);

	if (data->use_imag) {

		md_copy(N, zdims, tmp_z, zsrc, FL_SIZE);
		md_copy(N, wdims, tmp_w, wsrc, FL_SIZE);
	} else {

		md_real(N, zdims, tmp_z, zsrc);
		md_real(N, wdims, tmp_w, wsrc);
	}

	float* real_dst = md_alloc_sameplace(N, zdims, FL_SIZE, zdst);
	md_clear(N, zdims, real_dst, FL_SIZE);

	//use dest as tmp
	float* tmp1 = md_alloc_sameplace(N, zdims, FL_SIZE, zdst);

	for (int j = 0; j < Nw; j++) {

		md_pdf_gauss(N, zdims, tmp1, tmp_z, (mumin + j * dmu), data->sigma); //tmp1 = 1/sqrt(2pi sigma^2) *exp(-(z_ik-mu_j)^2/(2*sigma^2))

		long wpos[N];
		for (int i = 0; i < N; i++)
			wpos[i] = 0;
		wpos[data->idx_w] = j;

		const float* wtmp = tmp_w + md_calc_offset(data->N, data->wdom->strs, wpos) / FL_SIZE;

		md_mul2(N, zdims, zstrs, tmp1, wstrs, wtmp, zstrs, tmp1);
		md_add(N, zdims, real_dst, real_dst, tmp1);

		if (der1) {
			float scale = -(mumin + j * dmu);
			md_axpy(N, zdims, der_dz, scale, tmp1);
		}
	}

	md_free(tmp1);

	if (der1) {

		md_fmac(N, zdims, der_dz, real_dst, tmp_z); //data->dz = sum_j w_ij 1/sqrt(2pi sigma^2) * (z_ik - mu_j) *exp(-(z_ik-mu_j)^2/(2*sigma^2))
		md_smul(N, zdims, der_dz, der_dz, (- sqrtf(2. * M_PI) / data->sigma)); // zdst_ik = -1/sigma^2 sum_k (z_ik-mu_j) * w_ij * exp[-(z_ik-mu_j)²/(2*sigma²)]
	}

	if (der2)
		md_copy(N, zdims, der_z, tmp_z, FL_SIZE);

	md_free(tmp_z);
	md_free(tmp_w);

	md_smul(N, zdims, real_dst, real_dst, (sqrtf(2. * M_PI) * data->sigma)); // zdst_ik = -1/sigma^2 sum_k (z_ik-mu_j) * w_ij * exp[-(z_ik-mu_j)²/(2*sigma²)]

	if (data->use_imag)
		md_copy(data->zdom->N, data->zdom->dims, zdst, real_dst, FL_SIZE);
	else
		md_zcmpl_real(data->zdom->N, data->zdom->dims, zdst, real_dst);

	md_free(real_dst);
}

static void rbf_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	//dst_ik = sum_j src_ij * exp[-(z_ik-mu_j)^2/(s*sigma^2)]

	const auto data = CAST_DOWN(rbf_s, _data);
	float* der_z = data->z;

	int N = data->N;
	const long* zdims = data->zdom->dims;
	const long* wdims = data->wdom->dims;
	const long* zstrs = data->zdom->strs;
	const long* wstrs = data->wdom->strs;

	long Nw = wdims[data->idx_w];

	float mumin = data->Imin;
	float dmu = (data->Imax - data->Imin)/((float)Nw - 1.);

	float* real_src = md_alloc_sameplace(N, wdims, FL_SIZE, dst);
	if (data->use_imag)
		md_copy(N, wdims, real_src, src, FL_SIZE);
	else
		md_real(N, wdims, real_src, src);

	md_clear(N, zdims, dst, (data->use_imag) ? FL_SIZE : CFL_SIZE);

	float* real_dst = md_alloc_sameplace(N, zdims, FL_SIZE, dst);
	md_clear(N, zdims, real_dst, FL_SIZE);

	float* tmp1 = md_alloc_sameplace(N, zdims, FL_SIZE, dst);
	float* tmp2 = md_alloc_sameplace(N, zdims, FL_SIZE, dst);
	float* tmp3 = md_alloc_sameplace(N, zdims, FL_SIZE, dst);

	for (int j = 0; j < Nw; j++) {

		md_sadd(N, zdims, tmp1, der_z, -(mumin + j * dmu));
		md_mul(N, zdims, tmp2, tmp1, tmp1); // tmp2 = (z_ik-mu_j)²
		md_smul(N, zdims, tmp2, tmp2, (complex float)(-1. / (2 * data->sigma * data->sigma))); // tmp2 = -(z_ik-mu_j)²/(2*sigma²)
		md_exp(N, zdims, tmp2, tmp2); // tmp2 = exp[-(z_ik-mu_j)²/(2*sigma²)]

		long wpos[N];
		for (int i = 0; i < N; i++)
			wpos[i] = 0;
		wpos[data->idx_w] = j;

		const float* wtmp = real_src + md_calc_offset(N, wstrs, wpos) / FL_SIZE;

		md_copy2(N, zdims, zstrs, tmp3, wstrs, wtmp, FL_SIZE); // tmp3 = w_ik

		md_fmac(N, zdims, real_dst, tmp2, tmp3); // zdst_ik = sum_k w_ij * exp[-(z_ik-mu_j)²/(2*sigma²)]
	}

	md_free(tmp1);
	md_free(tmp2);
	md_free(tmp3);

	md_copy2(N, zdims, MD_STRIDES(N, zdims, data->use_imag ? FL_SIZE : CFL_SIZE), dst, MD_STRIDES(N, zdims, FL_SIZE), real_dst, FL_SIZE);
	md_free(real_dst);
	md_free(real_src);
}

static void rbf_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	//dst_ij = sum_k src_ik * exp[-(z_ik-mu_j)^2/(s*sigma^2)]

	const auto data = CAST_DOWN(rbf_s, _data);
	float* der_z = data->z;

	int N = data->N;
	const long* zdims = data->zdom->dims;
	const long* wdims = data->wdom->dims;
	const long* zstrs = data->zdom->strs;
	const long* wstrs = data->wdom->strs;

	long Nw = wdims[data->idx_w];
	float mumin = data->Imin;
	float dmu = (data->Imax - data->Imin)/((float)Nw - 1.);

	float* real_dst = md_alloc_sameplace(N, wdims, FL_SIZE, dst);
	md_clear(N, wdims, real_dst, FL_SIZE);

	float* real_src = md_alloc_sameplace(N, zdims, FL_SIZE, dst);
	if (data->use_imag)
		md_copy(N, zdims, real_src, src, FL_SIZE);
	else
		md_real(N, zdims, real_src, src);

	float* tmp1 = md_alloc_sameplace(N, zdims, FL_SIZE, dst);

	for (int j = 0; j < Nw; j++) {

		md_pdf_gauss(N, zdims, tmp1, der_z, (mumin + j * dmu), data->sigma);//tmp1 = 1/sqrt(2pi sigma^2) *exp(-(z_ik-mu_j)^2/(2*sigma^2))

		long wpos[N];
		for (int i = 0; i < N; i++)
			wpos[i] = 0;
		wpos[data->idx_w] = j;
		float* wtmp = real_dst + md_calc_offset(data->N, data->wdom->strs, wpos) / FL_SIZE;

		md_mul(N, zdims, tmp1, tmp1, real_src); // tmp1 = exp[-(z_ik-mu_j)²/(2*sigma²)] * phi_ik
		md_add2(N, zdims, wstrs, wtmp, wstrs, wtmp, zstrs, tmp1);
		//add is optimized for reductions -> change if fmac is optimized
	}
	md_free(real_src);
	md_free(tmp1);

	md_smul(N, wdims, real_dst, real_dst, (sqrtf(2. * M_PI) * data->sigma));

	if (data->use_imag)
		md_copy(N, wdims, dst, real_dst, FL_SIZE);
	else
		md_zcmpl_real(N, wdims, dst, real_dst);

	md_free(real_dst);

}

static void rbf_deradj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);


	const auto data = CAST_DOWN(rbf_s, _data);
	float* der_dz = data->dz;

	if (data->use_imag) {

		md_mul(data->zdom->N, data->zdom->dims, (float*)dst, (const float*)src, der_dz);
	} else {

		complex float* tmp = md_alloc_sameplace(data->N, data->zdom->dims, CFL_SIZE, dst);
		md_zcmpl_real(data->zdom->N, data->zdom->dims, tmp, der_dz);

		md_zrmul(data->N, data->zdom->dims, dst, src, tmp);

		md_free(tmp);
	}
}

static void rbf_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(rbf_s, _data);

	md_free(data->z);
	md_free(data->dz);

	iovec_free(data->zdom);
	iovec_free(data->wdom);

	xfree(data);
}

 /**
 * Create operator representing weighted combination of Gaussian radial basis functions (rbf)
 * \phi'_i = \sum_{j=1}^{N_w} w_{i, j} exp(-(z_i - \mu_j)^2 / (2 * \sigma^2))
 *
 * @param dims (Nf, Nb, Nw)
 * @param Imax max estimated filter response
 * @param Imin min estimated filter response
 * @param use_imag if true, imaginary part is considered as extra channel, i.e. Nf->2 * Nf, imaginary part is ignored
 *
 * Input tensors:
 * z		dims = (Nf, Nb)
 * w		dims = (Nf, Nw)
 *
 * Output tensors:
 * \phi'_i(z)	dims = (Nf, Nb)
 *
 * Note that Nb denotes the product of the parallel computable dimensions,
 * i.e. the number of different vectors z which is Nb = Nx*Ny*Nz*Nb
 */
const struct nlop_s* nlop_activation_rbf_create(const long dims[3], complex float Imax, complex float Imin, bool use_imag)
{
	PTR_ALLOC(struct rbf_s, data);
	SET_TYPEID(rbf_s, data);

	data->N = 3;
	data->idx_w = 2;
	data->use_imag = use_imag;

	long zdimsw[3];// {Nf, NB, 1 };
	long wdimsw[3];// {Nf, 1,  Nw};

	md_select_dims(3, 3, zdimsw, dims);
	md_select_dims(3, 5, wdimsw, dims);

	if (data->use_imag) {

		zdimsw[0] *= 2;
		wdimsw[0] *= 2;
	}

	data->zdom = iovec_create(3, zdimsw, FL_SIZE);
	data->wdom = iovec_create(3, wdimsw, FL_SIZE);

	data->z = NULL;
	data->dz = NULL;

	data->Imax = Imax;
	data->Imin = Imin;

	//The paper states \sigma = (Imax - Imin) / (Nw - 1)
	//However sigma is implemented differently, i.e. \sigma = (Imax - Imin) / (Nw)
	//C.f. https://github.com/VLOGroup/tensorflow-icg/blob/a11ad61d93d57c83f1af312b84a922e7612ec398/tensorflow/contrib/icg/kernels/activations.cu.cc#L123
	data->sigma = (float)(Imax - Imin) / (float)(dims[2]);

	long zdims[2] = {dims[0], dims[1]};
	long wdims[2] = {dims[0], dims[2]};

	long nl_odims[1][2];
	md_copy_dims(2, nl_odims[0], zdims);
	long nl_idims[2][2];
	md_copy_dims(2, nl_idims[0], zdims);
	md_copy_dims(2, nl_idims[1], wdims);

	auto result = nlop_generic_managed_create(1, 2, nl_odims, 2, 2, nl_idims, CAST_UP(PTR_PASS(data)), rbf_fun, (nlop_der_fun_t[2][1]){ { rbf_deradj1 }, { rbf_der2 } }, (nlop_der_fun_t[2][1]){ { rbf_deradj1 }, { rbf_adj2 } }, NULL, NULL, rbf_del, rbf_clear_der, NULL);
	return result;
}