/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */


#include <complex.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/wavelet.h"
#include "num/conv.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/blas.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"

#include "someops.h"

struct cdiag_s {

	linop_data_t base;

	unsigned int N;
	const long* dims;
	const long* strs;
	const long* dstrs;
	const complex float* diag;
#ifdef USE_CUDA
	const complex float* gpu_diag;
#endif
	bool rmul;
};

static void cdiag_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cdiag_s* data = CONTAINER_OF(_data, const struct cdiag_s, base);

	const complex float* diag = data->diag;
#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->gpu_diag)
			((struct cdiag_s*)data)->gpu_diag = md_gpu_move(data->N, data->dims, data->diag, CFL_SIZE);

		diag = data->gpu_diag;
	}
#endif
	(data->rmul ? md_zrmul2 : md_zmul2)(data->N, data->dims, data->strs, dst, data->strs, src, data->dstrs, diag);
}

static void cdiag_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct cdiag_s* data = CONTAINER_OF(_data, const struct cdiag_s, base);

	const complex float* diag = data->diag;
#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		if (NULL == data->gpu_diag)
			((struct cdiag_s*)data)->gpu_diag = md_gpu_move(data->N, data->dims, data->diag, CFL_SIZE);

		diag = data->gpu_diag;
	}
#endif
	(data->rmul ? md_zrmul2 : md_zmulc2)(data->N, data->dims, data->strs, dst, data->strs, src, data->dstrs, diag);
}

static void cdiag_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	cdiag_apply(_data, dst, src);
	cdiag_adjoint(_data, dst, dst);
}

static void cdiag_free(const linop_data_t* _data)
{
	const struct cdiag_s* data = CONTAINER_OF(_data, const struct cdiag_s, base);

#ifdef USE_CUDA
	md_free((void*)data->gpu_diag);
#endif
	free((void*)data->dims);
	free((void*)data->dstrs);
	free((void*)data->strs);
	free((void*)data);
}

static struct linop_s* linop_gdiag_create(unsigned int N, const long dims[N], unsigned int flags, const complex float* diag, bool rdiag)
{
	PTR_ALLOC(struct cdiag_s, data);

	data->rmul = rdiag;

	data->N = N;
	PTR_ALLOC(long[N], dims2);
	PTR_ALLOC(long[N], dstrs);
	PTR_ALLOC(long[N], strs);

	long ddims[N];
	md_select_dims(N, flags, ddims, dims);
	md_copy_dims(N, *dims2, dims);
	md_calc_strides(N, *strs, dims, CFL_SIZE);
	md_calc_strides(N, *dstrs, ddims, CFL_SIZE);

	data->dims = *PTR_PASS(dims2);
	data->strs = *PTR_PASS(strs);
	data->dstrs = *PTR_PASS(dstrs);
	data->diag = diag;	// make a copy?
#ifdef USE_CUDA
	data->gpu_diag = NULL;
#endif

	return linop_create(N, dims, N, dims, &PTR_PASS(data)->base, cdiag_apply, cdiag_adjoint, cdiag_normal, NULL, cdiag_free);
}



/**
 * Create a operator y = D x where D is a diagonal matrix
 *
 * @param N number of dimensions
 * @param dims input and output dimensions
 * @param flags bitmask specifiying the dimensions present in diag
 * @param diag diagonal matrix
 */
struct linop_s* linop_cdiag_create(unsigned int N, const long dims[N], unsigned int flags, const complex float* diag)
{
	return linop_gdiag_create(N, dims, flags, diag, false);
}


/**
 * Create a operator y = D x where D is a diagonal matrix
 *
 * @param N number of dimensions
 * @param dims input and output dimensions
 * @param flags bitmask specifiying the dimensions present in diag
 * @param diag diagonal matrix
 */
struct linop_s* linop_rdiag_create(unsigned int N, const long dims[N], unsigned int flags, const complex float* diag)
{
	return linop_gdiag_create(N, dims, flags, diag, true);
}



struct identity_data_s {

	linop_data_t base;

	const struct iovec_s* domain;
};

static void identity_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct iovec_s* domain = CONTAINER_OF(_data, const struct identity_data_s, base)->domain;

	md_copy2(domain->N, domain->dims, domain->strs, dst, domain->strs, src, CFL_SIZE);
}

static void identity_free(const linop_data_t* _data)
{	
	const struct identity_data_s* data = CONTAINER_OF(_data, const struct identity_data_s, base);

	iovec_free(data->domain);

	free((void*)data);
}

/**
 * Create an Identity linear operator: I x
 * @param N number of dimensions
 * @param dims dimensions of input (domain)
 */
struct linop_s* linop_identity_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct identity_data_s, data);

	data->domain = iovec_create(N, dims, CFL_SIZE);

	return linop_create(N, dims, N, dims, &PTR_PASS(data)->base, identity_apply, identity_apply, identity_apply, NULL, identity_free);
}


struct resize_op_s {

	linop_data_t base;

	unsigned int N;
	const long* out_dims;
	const long* in_dims;
};

static void resize_forward(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct resize_op_s* data = CONTAINER_OF(_data, const struct resize_op_s, base);

	md_resize_center(data->N, data->out_dims, dst, data->in_dims, src, CFL_SIZE);
}

static void resize_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct resize_op_s* data = CONTAINER_OF(_data, const struct resize_op_s, base);

	md_resize_center(data->N, data->in_dims, dst, data->out_dims, src, CFL_SIZE);
}

static void resize_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct resize_op_s* data = CONTAINER_OF(_data, const struct resize_op_s, base);

	complex float* tmp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);

	resize_forward(_data, tmp, src);
	resize_adjoint(_data, dst, tmp);

	md_free(tmp);
}

static void resize_free(const linop_data_t* _data)
{
	const struct resize_op_s* data = CONTAINER_OF(_data, const struct resize_op_s, base);

	free((void*)data->out_dims);
	free((void*)data->in_dims);

	free((void*)data);
}


/**
 * Create a resize linear operator: y = M x,
 * where M either crops or expands the the input dimensions to match the output dimensions.
 * Uses centered zero-padding and centered cropping
 *
 * @param N number of dimensions
 * @param out_dims output dimensions
 * @param in_dims input dimensions
 */
struct linop_s* linop_resize_create(unsigned int N, const long out_dims[N], const long in_dims[N])
{
	PTR_ALLOC(struct resize_op_s, data);

	data->N = N;
	data->out_dims = *TYPE_ALLOC(long[N]);
	data->in_dims = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, (long*)data->out_dims, out_dims);
	md_copy_dims(N, (long*)data->in_dims, in_dims);

	return linop_create(N, out_dims, N, in_dims, &PTR_PASS(data)->base, resize_forward, resize_adjoint, resize_normal, NULL, resize_free);
}


struct operator_matrix_s {

	linop_data_t base;

	const complex float* mat;
	const complex float* mat_conj;
	const complex float* mat_gram; // A^H A

	const struct iovec_s* mat_iovec;
	const struct iovec_s* mat_gram_iovec;
	const struct iovec_s* domain_iovec;
	const struct iovec_s* codomain_iovec;

	const long* max_dims;

	unsigned int K_dim;
	unsigned int K;
	unsigned int T_dim;
	unsigned int T;
};


/**
 * case 1: all singleton dimensions between T_dim and K_dim, all singleton dimensions after K_dim.
 *         then just apply standard matrix multiply
 */
static bool cgemm_forward_standard(const struct operator_matrix_s* data)
{
	long N = data->mat_iovec->N;
	long K_dim = data->K_dim;
	long T_dim = data->T_dim;

	bool use_cgemm = false;
	long dsum = 0;
	long csum = 0;

	//debug_printf(DP_DEBUG1, "T_dim = %d, K_dim = %d\n", T_dim, K_dim);
	//debug_print_dims(DP_DEBUG1, N, data->domain_iovec->dims);
	//debug_print_dims(DP_DEBUG1, N, data->codomain_iovec->dims);
	if (T_dim < K_dim) {
		
		for (int i = T_dim + 1; i < N; i++) {

			dsum += data->domain_iovec->dims[i] - 1;
			csum += data->codomain_iovec->dims[i] - 1;
			//debug_printf(DP_DEBUG1, "csum = %d, dsum = %d\n", csum, dsum);
		}
		// don't count K_dim
		dsum -= data->domain_iovec->dims[K_dim] - 1;

		if ((dsum + csum) == 0)
			use_cgemm = true;
	}

	//debug_printf(DP_DEBUG1, "use_cgemm = %d, dsum = %d, csum = %d\n", use_cgemm, dsum, csum);

	return use_cgemm;
}


static void linop_matrix_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct operator_matrix_s* data = CONTAINER_OF(_data, const struct operator_matrix_s, base);

	long N = data->mat_iovec->N;
	//debug_printf(DP_DEBUG1, "compute forward\n");

	md_clear2(N, data->codomain_iovec->dims, data->codomain_iovec->strs, dst, CFL_SIZE);

	// FIXME check all the cases where computation can be done with blas
	
	if (cgemm_forward_standard(data)) {

		long L = md_calc_size(data->T_dim, data->domain_iovec->dims);

		blas_cgemm('N', 'T', L, data->T, data->K, 1.,
				L, (const complex float (*)[])src,
				data->T, (const complex float (*)[])data->mat,
				0., L, (complex float (*)[])dst);

	} else {

		md_zfmac2(N, data->max_dims, data->codomain_iovec->strs, dst, data->domain_iovec->strs, src, data->mat_iovec->strs, data->mat);
	}
}

static void linop_matrix_apply_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct operator_matrix_s* data = CONTAINER_OF(_data, const struct operator_matrix_s, base);

	unsigned int N = data->mat_iovec->N;
	//debug_printf(DP_DEBUG1, "compute adjoint\n");

	md_clear2(N, data->domain_iovec->dims, data->domain_iovec->strs, dst, CFL_SIZE);

	// FIXME check all the cases where computation can be done with blas
	
	if (cgemm_forward_standard(data)) {

		long L = md_calc_size(data->T_dim, data->domain_iovec->dims);

		blas_cgemm('N', 'N', L, data->K, data->T, 1.,
				L, (const complex float (*)[])src,
				data->T, (const complex float (*)[])data->mat_conj,
				0., L, (complex float (*)[])dst);

	} else {

		md_zfmacc2(N, data->max_dims, data->domain_iovec->strs, dst, data->codomain_iovec->strs, src, data->mat_iovec->strs, data->mat);
	}
}

static void linop_matrix_apply_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct operator_matrix_s* data = CONTAINER_OF(_data, const struct operator_matrix_s, base);

	unsigned int N = data->mat_iovec->N;
	// FIXME check all the cases where computation can be done with blas
	
	//debug_printf(DP_DEBUG1, "compute normal\n");
	if (cgemm_forward_standard(data)) {

		long max_dims_gram[N];
		md_copy_dims(N, max_dims_gram, data->domain_iovec->dims);
		max_dims_gram[data->T_dim] = data->K;

		long tmp_dims[N];
		long tmp_str[N];
		md_copy_dims(N, tmp_dims, max_dims_gram);
		tmp_dims[data->K_dim] = 1;
		md_calc_strides(N, tmp_str, tmp_dims, CFL_SIZE);

		complex float* tmp = md_alloc_sameplace(N, data->domain_iovec->dims, CFL_SIZE, dst);

		md_clear(N, data->domain_iovec->dims, tmp, CFL_SIZE);
		md_zfmac2(N, max_dims_gram, tmp_str, tmp, data->domain_iovec->strs, src, data->mat_gram_iovec->strs, data->mat_gram);
		md_transpose(N, data->T_dim, data->K_dim, data->domain_iovec->dims, dst, tmp_dims, tmp, CFL_SIZE);

		md_free(tmp);

	} else {

		long L = md_calc_size(data->T_dim, data->domain_iovec->dims);

		blas_cgemm('N', 'T', L, data->K, data->K, 1.,
				L, (const complex float (*)[])src,
				data->K, (const complex float (*)[])data->mat_gram,
				0., L, (complex float (*)[])dst);
	}

}

static void linop_matrix_del(const linop_data_t* _data)
{
	const struct operator_matrix_s* data = CONTAINER_OF(_data, const struct operator_matrix_s, base);

	iovec_free(data->mat_iovec);
	iovec_free(data->mat_gram_iovec);
	iovec_free(data->domain_iovec);
	iovec_free(data->codomain_iovec);

	free((void*)data->max_dims);

	md_free((void*)data->mat);
	md_free((void*)data->mat_conj);
	md_free((void*)data->mat_gram);

	free((void*)data);
}


/**
 * Compute the Gram matrix, A^H A.
 * Stores the result in @param gram, which is allocated by the function
 * Returns: iovec_s corresponding to the gram matrix dimensions
 *
 * @param N number of dimensions
 * @param T_dim dimension corresponding to the rows of A
 * @param T number of rows of A (codomain)
 * @param K_dim dimension corresponding to the columns of A
 * @param K number of columns of A (domain)
 * @param gram store the result (allocated by this function)
 * @param matrix_dims dimensions of A
 * @param matrix matrix data
 */
const struct iovec_s* compute_gram_matrix(unsigned int N, unsigned int T_dim, unsigned int T, unsigned int K_dim, unsigned int K, complex float** gram, const long matrix_dims[N], const complex float* matrix)
{
	// FIXME this can certainly be simplfied...
	// Just be careful to consider the case where the data passed to the operator is a subset of a bigger array
	

	// B_dims = [T K 1]  or  [K T 1]
	// C_dims = [T 1 K]  or  [1 T K]
	// A_dims = [1 K K]  or  [K 1 K]
	// after: gram_dims = [1 K1 K2] --> [K2 K1 1]  or  [K1 1 K2] --> [K1 K2 1]

	long A_dims[N + 1];
	long B_dims[N + 1];
	long C_dims[N + 1];
	long fake_gram_dims[N + 1];

	long A_str[N + 1];
	long B_str[N + 1];
	long C_str[N + 1];
	long max_dims[N + 1];

	md_singleton_dims(N + 1, A_dims);
	md_singleton_dims(N + 1, B_dims);
	md_singleton_dims(N + 1, C_dims);
	md_singleton_dims(N + 1, fake_gram_dims);
	md_singleton_dims(N + 1, max_dims);

	A_dims[K_dim] = K;
	A_dims[N] = K;

	B_dims[T_dim] = T;
	B_dims[K_dim] = K;

	C_dims[T_dim] = T;
	C_dims[N] = K;

	max_dims[T_dim] = T;
	max_dims[K_dim] = K;
	max_dims[N] = K;

	fake_gram_dims[T_dim] = K;
	fake_gram_dims[K_dim] = K;

	md_calc_strides(N + 1, A_str, A_dims, CFL_SIZE);
	md_calc_strides(N + 1, B_str, B_dims, CFL_SIZE);
	md_calc_strides(N + 1, C_str, C_dims, CFL_SIZE);

	complex float* tmpA = md_alloc_sameplace(N + 1 , A_dims, CFL_SIZE, matrix);
	complex float* tmpB = md_alloc_sameplace(N + 1, B_dims, CFL_SIZE, matrix);
	complex float* tmpC = md_alloc_sameplace(N + 1, C_dims, CFL_SIZE, matrix);

	md_copy(N, matrix_dims, tmpB, matrix, CFL_SIZE);
	//md_copy(N, matrix_dims, tmpC, matrix, CFL_SIZE);

	md_transpose(N + 1, K_dim, N, C_dims, tmpC, B_dims, tmpB, CFL_SIZE);
	md_clear(N + 1, A_dims, tmpA, CFL_SIZE);
	md_zfmacc2(N + 1, max_dims, A_str, tmpA, B_str, tmpB, C_str, tmpC);

	*gram = md_alloc_sameplace(N, fake_gram_dims, CFL_SIZE, matrix);
	md_transpose(N + 1, T_dim, N, fake_gram_dims, *gram, A_dims, tmpA, CFL_SIZE); 


	const struct iovec_s* s =  iovec_create(N, fake_gram_dims, CFL_SIZE);

	md_free(tmpA);
	md_free(tmpB);
	md_free(tmpC);

	return s;
}


/**
 * Operator interface for a true matrix:
 * out = mat * in
 * in:	[x x x x 1 x x K x x]
 * mat:	[x x x x T x x K x x]
 * out:	[x x x x T x x 1 x x]
 * where the x's are arbitrary dimensions and T and K may be transposed
 *
 * use this interface if K == 1 or T == 1
 *
 * @param N number of dimensions
 * @param out_dims output dimensions after applying the matrix (codomain)
 * @param in_dims input dimensions to apply the matrix (domain)
 * @param T_dim dimension corresponding to the rows of A
 * @param K_dim dimension corresponding to the columns of A
 * @param matrix matrix data
 */
struct linop_s* linop_matrix_altcreate(unsigned int N, const long out_dims[N], const long in_dims[N], const unsigned int T_dim, const unsigned int K_dim, const complex float* matrix)
{
	long matrix_dims[N];
	md_singleton_dims(N, matrix_dims);

	matrix_dims[K_dim] = in_dims[K_dim];
	matrix_dims[T_dim] = out_dims[T_dim];

	unsigned int T = out_dims[T_dim];
	unsigned int K = in_dims[K_dim];

	PTR_ALLOC(long[N], max_dims);

	for (unsigned int i = 0; i < N; i++) {

		if ((in_dims[i] > 1) && (out_dims[i] == 1)) {

			(*max_dims)[i] = in_dims[i];
		}
		else if ((in_dims[i] == 1) && (out_dims[i] > 1)) {

			(*max_dims)[i] = out_dims[i];
		}
		else {

			assert(in_dims[i] == out_dims[i]);

			(*max_dims)[i] = in_dims[i];
		}
	}

	complex float* mat = md_alloc_sameplace(N, matrix_dims, CFL_SIZE, matrix);
	complex float* matc = md_alloc_sameplace(N, matrix_dims, CFL_SIZE, matrix);

	md_copy(N, matrix_dims, mat, matrix, CFL_SIZE);
	md_zconj(N, matrix_dims, matc, mat);

	complex float* gram = NULL;
	const struct iovec_s* gram_iovec = compute_gram_matrix(N, T_dim, T, K_dim, K, &gram, matrix_dims, matrix);

	PTR_ALLOC(struct operator_matrix_s, data);

	data->mat_iovec = iovec_create(N, matrix_dims, CFL_SIZE);
	data->mat_gram_iovec = gram_iovec;

	data->max_dims = *max_dims;

	data->mat = mat;
	data->mat_conj = matc;
	data->mat_gram = gram;

	data->K_dim = K_dim;
	data->T_dim = T_dim;
	data->K = K;
	data->T = T;

	data->domain_iovec = iovec_create(N, in_dims, CFL_SIZE);
	data->codomain_iovec = iovec_create(N, out_dims, CFL_SIZE);

	return linop_create(N, out_dims, N, in_dims, &PTR_PASS(data)->base, linop_matrix_apply, linop_matrix_apply_adjoint, linop_matrix_apply_normal, NULL, linop_matrix_del);
}


/**
 * Operator interface for a true matrix:
 * out = mat * in
 * in:	[x x x x 1 x x K x x]
 * mat:	[x x x x T x x K x x]
 * out:	[x x x x T x x 1 x x]
 * where the x's are arbitrary dimensions and T and K may be transposed
 *
 * FIXME if K == 1 or T == 1, use the linop_matrixalt_create
 *
 * @param N number of dimensions
 * @param out_dims output dimensions after applying the matrix (codomain)
 * @param in_dims input dimensions to apply the matrix (domain)
 * @param matrix_dims dimensions of the matrix
 * @param matrix matrix data
 */
struct linop_s* linop_matrix_create(unsigned int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const complex float* matrix)
{
	//FIXME can auto-compute some of the dimensions, use flags, etc...
	//FIXME check that dimensions are consistent
	
	UNUSED(matrix_dims);

	unsigned int K_dim = N + 1;
	unsigned int T_dim = N + 1;

	for (unsigned int i = 0; i < N; i++) {

		if ((in_dims[i] > 1) && (out_dims[i] == 1)) {

			K_dim = i;
		}
		else if ((in_dims[i] == 1) && (out_dims[i] > 1)) {

			T_dim = i;
		}
		else {

			assert(in_dims[i] == out_dims[i]);
		}
	}

	assert((K_dim < (N + 1)) && (T_dim < (N + 1))); // FIXME what if K_dim or T_dim == 1?

	return linop_matrix_altcreate(N, out_dims, in_dims, T_dim, K_dim, matrix);
}


/**
 * Efficiently chain two matrix linops by multiplying the actual matrices together.
 * Stores a copy of the new matrix.
 * Returns: C = B A
 *
 * @param a first matrix (applied to input)
 * @param b second matrix (applied to output of first matrix)
 */
struct linop_s* linop_matrix_chain(const struct linop_s* a, const struct linop_s* b)
{
	const struct operator_matrix_s* a_data = linop_get_data(a);
	const struct operator_matrix_s* b_data = linop_get_data(b);

	// check compatibility
	assert(linop_codomain(a)->N == linop_domain(b)->N);
	assert(md_calc_size(linop_codomain(a)->N, linop_codomain(a)->dims) == md_calc_size(linop_domain(b)->N, linop_domain(b)->dims));
	assert(a_data->K_dim != b_data->T_dim); // FIXME error for now -- need to deal with this specially.
	assert((a_data->T_dim == b_data->K_dim) && (a_data->T == b_data->K));

	unsigned int N = linop_domain(a)->N;

	long max_dims[N];

	md_singleton_dims(N, max_dims);
	max_dims[a_data->T_dim] = a_data->T;
	max_dims[a_data->K_dim] = a_data->K;
	max_dims[b_data->T_dim] = b_data->T;

	long matrix_dims[N];
	long matrix_strs[N];

	md_select_dims(N, ~MD_BIT(a_data->T_dim), matrix_dims, max_dims);
	md_calc_strides(N, matrix_strs, matrix_dims, CFL_SIZE);

	complex float* matrix = md_alloc_sameplace(N, matrix_dims, CFL_SIZE, a_data->mat);

	md_clear(N, matrix_dims, matrix, CFL_SIZE);
	md_zfmac2(N, max_dims, matrix_strs, matrix, a_data->mat_iovec->strs, a_data->mat, b_data->mat_iovec->strs, b_data->mat);

	struct linop_s* c = linop_matrix_create(N, linop_codomain(b)->dims, linop_domain(a)->dims, matrix_dims, matrix);

	md_free(matrix);

	return c;
}


struct fft_linop_s {

	linop_data_t base;

	const struct operator_s* frw;
	const struct operator_s* adj;

	bool center;
	float nscale;
	
	int N;
	long* dims;
	long* strs;
};

static void fft_linop_apply(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct fft_linop_s* data = CONTAINER_OF(_data, const struct fft_linop_s, base);

	if (in != out)
		md_copy2(data->N, data->dims, data->strs, out, data->strs, in, CFL_SIZE);

	operator_apply(data->frw, data->N, data->dims, out, data->N, data->dims, out);
}

static void fft_linop_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct fft_linop_s* data = CONTAINER_OF(_data, const struct fft_linop_s, base);

	if (in != out)
		md_copy2(data->N, data->dims, data->strs, out, data->strs, in, CFL_SIZE);

	operator_apply(data->adj, data->N, data->dims, out, data->N, data->dims, out);
}

static void fft_linop_free(const linop_data_t* _data)
{
	const struct fft_linop_s* data = CONTAINER_OF(_data, const struct fft_linop_s, base);

	fft_free(data->frw);
	fft_free(data->adj);

	free(data->dims);
	free(data->strs);

	free((void*)data);
}

static void fft_linop_normal(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct fft_linop_s* data = CONTAINER_OF(_data, const struct fft_linop_s, base);

	if (data->center)
		md_copy(data->N, data->dims, out, in, CFL_SIZE);
	else
		md_zsmul(data->N, data->dims, out, in, data->nscale);
}


static struct linop_s* linop_fft_create_priv(int N, const long dims[N], unsigned int flags, bool forward, bool center)
{
	bool gpu = true; // FIXME
#ifdef USE_CUDA
	md_alloc_fun_t alloc = (gpu ? md_alloc_gpu : md_alloc);
#else
	UNUSED(gpu);
	md_alloc_fun_t alloc = md_alloc;
#endif

	// FIXME: we allocate only to communicate the gpu flag
	// and that need in-place plans
	complex float* tmp = alloc(N, dims, CFL_SIZE);
	const struct operator_s* plan = fft_create(N, dims, flags, tmp, tmp, false);
	const struct operator_s* iplan = fft_create(N, dims, flags, tmp, tmp, true);
	md_free(tmp);

	PTR_ALLOC(struct fft_linop_s, data);

	data->frw = plan;
	data->adj = iplan;
	data->N = N;

	data->center = center;

	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);

	data->strs = *TYPE_ALLOC(long[N]);
	md_calc_strides(N, data->strs, data->dims, CFL_SIZE);

	long fft_dims[N];
	md_select_dims(N, flags, fft_dims, dims);
	data->nscale = (float)md_calc_size(N, fft_dims);

	lop_fun_t apply = forward ? fft_linop_apply : fft_linop_adjoint;
	lop_fun_t adjoint = forward ? fft_linop_adjoint : fft_linop_apply;

	struct linop_s* lop =  linop_create(N, dims, N, dims, &PTR_PASS(data)->base, apply, adjoint, fft_linop_normal, NULL, fft_linop_free);

	if (center) {

		// FIXME: should only allocate flagged dims

		complex float* fftmod_mat = md_alloc(N, dims, CFL_SIZE);
		complex float* fftmodk_mat = md_alloc(N, dims, CFL_SIZE);

		// we need fftmodk only because we want to apply scaling only once

		complex float one[1] = { 1. };
		md_fill(N, dims, fftmod_mat, one, CFL_SIZE);
		fftmod(N, dims, flags, fftmodk_mat, fftmod_mat);
		fftscale(N, dims, flags, fftmod_mat, fftmodk_mat);

		struct linop_s* mod = linop_cdiag_create(N, dims, ~0u, fftmod_mat);
		struct linop_s* modk = linop_cdiag_create(N, dims, ~0u, fftmodk_mat);

		struct linop_s* tmp = linop_chain(mod, lop);
		tmp = linop_chain(tmp, modk);

		linop_free(lop);
		linop_free(mod);
		linop_free(modk);

		lop = tmp;
	}

	return lop;
}


/**
 * Uncentered forward Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 * @param gpu use gpu
 */
struct linop_s* linop_fft_create(int N, const long dims[N], unsigned int flags)
{
	return linop_fft_create_priv(N, dims, flags, true, false);
}


/**
 * Uncentered inverse Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_ifft_create(int N, const long dims[N], unsigned int flags)
{
	return linop_fft_create_priv(N, dims, flags, false, false);
}


/**
 * Centered forward Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_fftc_create(int N, const long dims[N], unsigned int flags)
{
	return linop_fft_create_priv(N, dims, flags, true, true);
}


/**
 * Centered inverse Fourier transform linear operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_ifftc_create(int N, const long dims[N], unsigned int flags)
{
	return linop_fft_create_priv(N, dims, flags, false, true);
}




struct linop_cdf97_s {

	linop_data_t base;

	unsigned int N;
	const long* dims;
	unsigned int flags;
};

static void linop_cdf97_apply(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct linop_cdf97_s* data = CONTAINER_OF(_data, struct linop_cdf97_s, base);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
	md_cdf97z(data->N, data->dims, data->flags, out);
}

static void linop_cdf97_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct linop_cdf97_s* data = CONTAINER_OF(_data, struct linop_cdf97_s, base);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
	md_icdf97z(data->N, data->dims, data->flags, out);
}

static void linop_cdf97_normal(const linop_data_t* _data, complex float* out, const complex float* in)
{
	const struct linop_cdf97_s* data = CONTAINER_OF(_data, struct linop_cdf97_s, base);

	md_copy(data->N, data->dims, out, in, CFL_SIZE);
}

static void linop_cdf97_free(const linop_data_t* _data)
{
	const struct linop_cdf97_s* data = CONTAINER_OF(_data, struct linop_cdf97_s, base);

	free((void*)data->dims);

	free((void*)data);
}



/**
 * Wavelet CFD9/7 transform operator
 *
 * @param N number of dimensions
 * @param dims dimensions of input
 * @param flags bitmask of the dimensions to apply the Fourier transform
 */
struct linop_s* linop_cdf97_create(int N, const long dims[N], unsigned int flags)
{
	PTR_ALLOC(struct linop_cdf97_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *ndims;
	data->flags = flags;

	return linop_create(N, dims, N, dims, &data->base, linop_cdf97_apply, linop_cdf97_adjoint, linop_cdf97_normal, NULL, linop_cdf97_free);
}



struct conv_data_s {

	linop_data_t base;

	struct conv_plan* plan;
};

static void linop_conv_forward(const linop_data_t* _data, complex float* out, const complex float* in)
{
	struct conv_data_s* data = CONTAINER_OF(_data, struct conv_data_s, base);

	conv_exec(data->plan, out, in);
}

static void linop_conv_adjoint(const linop_data_t* _data, complex float* out, const complex float* in)
{
	struct conv_data_s* data = CONTAINER_OF(_data, struct conv_data_s, base);

	conv_adjoint(data->plan, out, in);
}

static void linop_conv_free(const linop_data_t* _data)
{
	struct conv_data_s* data = CONTAINER_OF(_data, struct conv_data_s, base);

	conv_free(data->plan);

	free(data);
}


/**
 * Convolution operator
 *
 * @param N number of dimensions
 * @param flags bitmask of the dimensions to apply convolution
 * @param ctype
 * @param cmode
 * @param odims output dimensions
 * @param idims input dimensions 
 * @param kdims kernel dimensions
 * @param krn convolution kernel 
 */
struct linop_s* linop_conv_create(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[N],
                const long idims[N], const long kdims[N], const complex float* krn)
{
	PTR_ALLOC(struct conv_data_s, data);

	data->plan = conv_plan(N, flags, ctype, cmode, odims, idims, kdims, krn);

	return linop_create(N, odims, N, idims, &PTR_PASS(data)->base, linop_conv_forward, linop_conv_adjoint, NULL, NULL, linop_conv_free);
}


