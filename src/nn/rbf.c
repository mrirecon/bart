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
	const struct iovec_s* dom;	// (Nf, Nb, Nw) index convention: (i, k, j)
	const struct iovec_s* zdom;	// (Nf, Nb, 1 )
	const struct iovec_s* wdom;	// (Nf, 1,  Nw)
	const struct iovec_s* mudom;	// (1,  1,  Nw)

	float Imax;
	float Imin;
	float sigma;
};

DEF_TYPEID(rbf_s);


#if 0
static void rbf_initialize(struct rbf_s* data, const complex float* arg, bool der1)
{
	if (NULL == data->w)
		data->w = md_alloc_sameplace(data->N, data->wdom->dims, FL_SIZE, arg);

	if (NULL == data->z)
		data->z = md_alloc_sameplace(data->N, data->zdom->dims, FL_SIZE, arg);

	if (der1 && (NULL == data->dz))
		data->dz = md_alloc_sameplace(data->N, data->zdom->dims, FL_SIZE, arg);

	if (der1)
		md_clear(data->N, data->zdom->dims, data->dz, FL_SIZE);

	if (!der1 &&(NULL != data->dz)){
		md_free(data->dz);
		data->dz = NULL;
	}
}

static void rbf_set_opts(const nlop_data_t* _data, const struct op_options_s* opts)
{
	const auto data = CAST_DOWN(rbf_s, _data);

	if(op_options_is_set_io(opts, 0, 0, OP_APP_CLEAR_DER)){

		md_free(data->dz);
		data->dz = NULL;
	}
	if(op_options_is_set_io(opts, 0, 1, OP_APP_CLEAR_DER)){

		md_free(data->z);
		md_free(data->w);
		data->z = NULL;
		data->w = NULL;
	}
}

#endif


static void rbf_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	//dst_ik = sum_j w_ij * exp[-(z_ik-mu_j)^2/(s*sigma^2)]
	//data->dz_ik = sum_j (mu_j - z_ik)/sigma^2 * w_ij * exp[-(z_ik-mu_j)^2/(s*sigma^2)]


	assert(3 == N);
	const auto data = CAST_DOWN(rbf_s, _data);

	complex float* zdst = args[0];
	const complex float* zsrc = args[1];
	const complex float* wsrc = args[2];

	bool der1 = nlop_der_requested(_data, 0, 0);
	bool der2 = nlop_der_requested(_data, 1, 0);

	nlop_data_der_alloc_memory(_data, zdst);

	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	float* der_z = der_data[0];
	float* der_dz = der_data[1];
	if(NULL != der_dz)
		md_clear(2, data->zdom->dims, der_dz, FL_SIZE);

	float* tmp_w = md_alloc_sameplace(data->N, data->wdom->dims, FL_SIZE, zdst);
	float* tmp_z = md_alloc_sameplace(data->N, data->zdom->dims, FL_SIZE, zdst);

	long Nw = data->dom->dims[2];
	float mumin = data->Imin;
	float dmu = (data->Imax - data->Imin)/((float)Nw - 1.);

	md_real(data->zdom->N, data->zdom->dims, tmp_z, zsrc);
	md_real(data->zdom->N, data->wdom->dims, tmp_w, wsrc);

	float* real_dst = md_alloc_sameplace(data->N, data->zdom->dims, FL_SIZE, zdst);
	md_clear(data->N, data->zdom->dims, real_dst, FL_SIZE);

	//use dest as tmp
	float* tmp1 = (float*)zdst;
	float* tmp2 = (float*)zdst + md_calc_size(data->N, data->zdom->dims);


	for (int j = 0; j < Nw; j++) {

		md_pdf_gauss(2, data->zdom->dims, tmp1, tmp_z, (mumin + j * dmu), data->sigma); //tmp1 = 1/sqrt(2pi sigma^2) *exp(-(z_ik-mu_j)^2/(2*sigma^2))

		long wpos[3] = {0, 0, j};
		const float* wtmp = tmp_w + md_calc_offset(data->N, data->wdom->strs, wpos) / FL_SIZE;

		md_mul2(2, data->dom->dims, data->zdom->strs, tmp1, data->wdom->strs, wtmp, data->zdom->strs, tmp1);
		md_add(2, data->zdom->dims, real_dst, real_dst, tmp1);

		if (der1) {
			float scale = -(mumin + j * dmu);
			md_copy(1, MAKE_ARRAY(1l), tmp2, &scale, FL_SIZE);
			md_fmac2(2, data->zdom->dims, data->zdom->strs, der_dz, data->zdom->strs, tmp1, MAKE_ARRAY(0l, 0l), tmp2);
		}
	}

	if (der1) {

		md_fmac(3, data->zdom->dims, der_dz, real_dst, tmp_z); //data->dz = sum_j w_ij 1/sqrt(2pi sigma^2) * (z_ik - mu_j) *exp(-(z_ik-mu_j)^2/(2*sigma^2))
		md_smul(3, data->zdom->dims, der_dz, der_dz, (- sqrtf(2. * M_PI) / data->sigma)); // zdst_ik = -1/sigma^2 sum_k (z_ik-mu_j) * w_ij * exp[-(z_ik-mu_j)²/(2*sigma²)]
	}

	if (der2)
		md_copy(data->N, data->zdom->dims, der_z, tmp_z, FL_SIZE);

	md_free(tmp_z);
	md_free(tmp_w);

	md_smul(3, data->zdom->dims, real_dst, real_dst, (sqrtf(2. * M_PI) * data->sigma)); // zdst_ik = -1/sigma^2 sum_k (z_ik-mu_j) * w_ij * exp[-(z_ik-mu_j)²/(2*sigma²)]

	md_zcmpl_real(data->zdom->N, data->zdom->dims, zdst, real_dst);
	md_free(real_dst);
}

static void rbf_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	float* der_z = der_data[0];
	//float* der_dz = der_data[1];


	//dst_ik = sum_j src_ij * exp[-(z_ik-mu_j)^2/(s*sigma^2)]
	const auto data = CAST_DOWN(rbf_s, _data);

	long Nw = data->dom->dims[2];
	float mumin = data->Imin;
	float dmu = (data->Imax - data->Imin)/((float)Nw - 1.);

	float* real_src = md_alloc_sameplace(data->N, data->wdom->dims, FL_SIZE, dst);
	md_copy2(data->wdom->N, data->wdom->dims, MD_STRIDES(data->wdom->N, data->wdom->dims, FL_SIZE), real_src, MD_STRIDES(data->wdom->N, data->wdom->dims, CFL_SIZE), src, FL_SIZE);

	md_clear(data->N, data->zdom->dims, dst, CFL_SIZE);

	float* real_dst = md_alloc_sameplace(data->N, data->zdom->dims, FL_SIZE, dst);
	md_clear(data->N, data->zdom->dims, real_dst, FL_SIZE);

	float* tmp1 = md_alloc_sameplace(2, data->zdom->dims, FL_SIZE, dst);
	float* tmp2 = md_alloc_sameplace(2, data->zdom->dims, FL_SIZE, dst);
	float* tmp3 = md_alloc_sameplace(2, data->zdom->dims, FL_SIZE, dst);

	for (int j = 0; j < Nw; j++) {

		md_sadd(2, data->zdom->dims, tmp1, der_z, -(mumin + j * dmu));
		md_mul(2, data->dom->dims, tmp2, tmp1, tmp1); // tmp2 = (z_ik-mu_j)²
		md_smul(2, data->dom->dims, tmp2, tmp2, (complex float)(-1. / (2 * data->sigma * data->sigma))); // tmp2 = -(z_ik-mu_j)²/(2*sigma²)
		md_exp(2, data->dom->dims, tmp2, tmp2); // tmp2 = exp[-(z_ik-mu_j)²/(2*sigma²)]

		long wpos[3] = {0, 0, j};
		const float* wtmp = real_src + md_calc_offset(data->N, data->wdom->strs, wpos) / FL_SIZE;
		md_copy2(2, data->dom->dims, data->zdom->strs, tmp3, data->wdom->strs, wtmp, FL_SIZE); // tmp3 = w_ik

		md_fmac(2, data->dom->dims, real_dst, tmp2, tmp3); // zdst_ik = sum_k w_ij * exp[-(z_ik-mu_j)²/(2*sigma²)]
	}

	md_free(tmp1);
	md_free(tmp2);
	md_free(tmp3);

	md_copy2(data->zdom->N, data->zdom->dims, MD_STRIDES(data->zdom->N, data->zdom->dims, CFL_SIZE), dst, MD_STRIDES(data->zdom->N, data->zdom->dims, FL_SIZE), real_dst, FL_SIZE);
	md_free(real_dst);
	md_free(real_src);


}

static void rbf_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	float* der_z = der_data[0];
	//float* der_dz = der_data[1];

	//dst_ij = sum_k src_ik * exp[-(z_ik-mu_j)^2/(s*sigma^2)]
	const auto data = CAST_DOWN(rbf_s, _data);

	long Nw = data->dom->dims[2];
	float mumin = data->Imin;
	float dmu = (data->Imax - data->Imin)/((float)Nw - 1.);

	float* real_dst = md_alloc_sameplace(data->N, data->wdom->dims, FL_SIZE, dst);
	md_clear(data->N, data->wdom->dims, real_dst, FL_SIZE);

	float* real_src = md_alloc_sameplace(2, data->zdom->dims, FL_SIZE, dst);
	md_real(data->zdom->N, data->zdom->dims, real_src, src);

	float* tmp1 = md_alloc_sameplace(2, data->zdom->dims, FL_SIZE, dst);

	for (int j = 0; j < Nw; j++) {

		md_pdf_gauss(2, data->zdom->dims, tmp1, der_z, (mumin + j * dmu), data->sigma);//tmp1 = 1/sqrt(2pi sigma^2) *exp(-(z_ik-mu_j)^2/(2*sigma^2))
		long wpos[3] = {0, 0, j};
		float* wtmp = real_dst + md_calc_offset(data->N, data->wdom->strs, wpos) / FL_SIZE;

		md_mul(2, data->dom->dims, tmp1, tmp1, real_src); // tmp1 = exp[-(z_ik-mu_j)²/(2*sigma²)] * phi_ik
		md_add2(2, data->dom->dims, data->wdom->strs, wtmp, data->wdom->strs, wtmp, data->zdom->strs, tmp1);
		//add is optimized for reductions -> change if fmac is optimized
	}
	md_free(real_src);
	md_free(tmp1);

	md_smul(3, data->wdom->dims, real_dst, real_dst, (sqrtf(2. * M_PI) * data->sigma));

	md_zcmpl_real(data->wdom->N, data->wdom->dims, dst, real_dst);
	md_free(real_dst);

}

static void rbf_der1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);


	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	//float* der_z = der_data[0];
	float* der_dz = der_data[1];

	const auto data = CAST_DOWN(rbf_s, _data);

	complex float* tmp = md_alloc_sameplace(data->N, data->zdom->dims, CFL_SIZE, dst);
	md_zcmpl_real(data->zdom->N, data->zdom->dims, tmp, der_dz);

	md_zrmul(data->N, data->zdom->dims, dst, src, tmp);

	md_free(tmp);
}

static void rbf_adj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(rbf_s, _data);


	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	//float* der_z = der_data[0];
	float* der_dz = der_data[1];

	complex float* tmp = md_alloc_sameplace(data->N, data->zdom->dims, CFL_SIZE, dst);
	md_zcmpl_real(data->zdom->N, data->zdom->dims, tmp, der_dz);

	md_zrmul(data->N, data->zdom->dims, dst, src, tmp);

	md_free(tmp);
}

static void rbf_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(rbf_s, _data);

	iovec_free(data->dom);
	iovec_free(data->zdom);
	iovec_free(data->wdom);
	iovec_free(data->mudom);

	xfree(data);
}

 /**
 * Create operator representing weighted combination of Gaussian radial basis functions (rbf)
 * \phi'_i = \sum_{j=1}^{N_w} w_{i, j} exp(-(z_i - \mu_j)^2 / (2 * \sigma^2))
 *
 * @param dims (Nf, Nb, Nw)
 * @param Imax max estimated filter response
 * @param Imin min estimated filter response
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
const struct nlop_s* nlop_activation_rbf_create(const long dims[3], complex float Imax, complex float Imin)
{
	PTR_ALLOC(struct rbf_s, data);
	SET_TYPEID(rbf_s, data);

	data->N = 3;

	long zdimsw[3];// {Nf, NB, 1 };
	long wdimsw[3];// {Nf, 1,  Nw};
	long mudims[3];// {1,  1,  Nw};

	md_select_dims(3, 3, zdimsw, dims);
	md_select_dims(3, 5, wdimsw, dims);
	md_select_dims(3, 4, mudims, dims);

	data->dom = iovec_create(3, dims, FL_SIZE);
	data->zdom = iovec_create(3, zdimsw, FL_SIZE);
	data->wdom = iovec_create(3, wdimsw, FL_SIZE);
	data->mudom = iovec_create(3, mudims, FL_SIZE);

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

	struct nlop_der_array_s* der_arrays[2];
	der_arrays[0] = nlop_der_array_create(3, zdimsw, FL_SIZE, 2, 1, (bool[2][1]){{ false }, { true }});
	der_arrays[1] = nlop_der_array_create(3, zdimsw, FL_SIZE, 2, 1, (bool[2][1]){{ true }, { false }});

	auto result = nlop_generic_managed_create(1, 2, nl_odims, 2, 2, nl_idims, CAST_UP(PTR_PASS(data)), rbf_fun, (nlop_der_fun_t[2][1]){ { rbf_der1 }, { rbf_der2 } }, (nlop_der_fun_t[2][1]){ { rbf_adj1 }, { rbf_adj2 } }, NULL, NULL, rbf_del, 2, der_arrays, NULL);
	return result;
}