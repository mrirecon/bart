/* Copyright 2021. Uecker Lab. University Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/multiplace.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"

#include "const.h"


struct const_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	const long* strs;
	const complex float* xn_ref;

	struct multiplace_array_s* xn_cop;

	bool copied;
};

DEF_TYPEID(const_s);

static void const_fun(const nlop_data_t* _data, int N, complex float** dst)
{
	UNUSED(N);
	const auto data = CAST_DOWN(const_s, _data);

	const complex float* ref = data->copied ? multiplace_read(data->xn_cop, NULL) : data->xn_ref;

	md_copy2(data->N, data->dims, MD_STRIDES(data->N, data->dims, CFL_SIZE), dst[0], data->strs, ref, CFL_SIZE);
}

static void const_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(const_s, _data);

	multiplace_free(data->xn_cop);

	xfree(data->dims);
	xfree(data->strs);
	xfree(data);
}


/**
 * Create operator with constant output (zero inputs, one output)
 * Strides are only applied on the input
 * @param N #dimensions
 * @param dims dimensions
 * @param strs in-strides
 * @param copy decide if const in is copied in operator
 * @param in reference to constant input array
 */
struct nlop_s* nlop_const_create2(int N, const long dims[N], const long strs[N], bool copy, const complex float* in)
{
	PTR_ALLOC(struct const_s, data);
	SET_TYPEID(const_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);

	data->copied = copy;

	PTR_ALLOC(long[N], nstrs);

	if (copy) {

		data->xn_cop = multiplace_move2(N, dims, strs, CFL_SIZE, in);
		data->xn_ref = NULL;

		md_calc_strides(N, *nstrs, dims, CFL_SIZE);

	} else {

		data->xn_cop = NULL;
		data->xn_ref = in;

		md_copy_dims(N, *nstrs, strs);
	}

	data->strs = *PTR_PASS(nstrs);


	long ostrs[N];
	md_calc_strides(N, ostrs, dims, CFL_SIZE);

	long tdims[1][N];
	md_copy_dims(N, tdims[0], dims);

	return nlop_generic_create(1, N, tdims, 0, 0, NULL, CAST_UP(PTR_PASS(data)), const_fun, NULL, NULL, NULL,NULL, const_del);
}


/**
 * Create operator with constant output (zero inputs, one output)
 * @param N #dimensions
 * @param dims dimensions
 * @param copy decide if const in is copied in operator
 * @param in reference to constant input array
 */
struct nlop_s* nlop_const_create(int N, const long dims[N], bool copy, const complex float* in)
{
	return nlop_const_create2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), copy, in);
}


/**
 * Chain operator with a constant operator
 * @param a operator whose input should be set constant
 * @param i index which should be set constant
 * @param N #dimensions of input array
 * @param dims dimensions of input array
 * @param strs strides of input array
 * @param copy decide if const in is copied in operator
 * @param in pointer to input array
 */
struct nlop_s* nlop_set_input_const2(const struct nlop_s* a, int i, int N, const long dims[N], const long strs[N], bool copy, const complex float* in)
{
	int ai = nlop_get_nr_in_args(a);

	assert(i < ai);

	auto iov = nlop_generic_domain(a, i);

	int N_min = (N < (int)(iov->N)) ? N : (int)(iov->N);
	int N_max = (N > (int)(iov->N)) ? N : (int)(iov->N);
	long ndims[N_max];
	long nstrs[N_max];

	md_singleton_dims(N_max, ndims);
	md_singleton_strides(N_max, nstrs);
	md_copy_dims(N_min, ndims, dims);
	md_copy_strides(N_min, nstrs, strs);

	for (int i = N_min; i < iov->N; i++)
		assert(1 == iov->dims[i]);

	for (int i = N_min; i < N; i++)
		assert(1 == dims[i]);

	struct nlop_s* nlop_const = nlop_const_create2(iov->N, ndims, nstrs, copy, in);
	struct nlop_s* result = nlop_chain2(nlop_const, 0,  a,  i);

	nlop_free(nlop_const);

	return result;
}


/**
 * Chain operator with a constant operator
 * @param a operator whose input should be set constant
 * @param i index which should be set constant
 * @param N #dimensions of input array
 * @param dims dimensions of input array
 * @param copy decide if const in is copied in operator
 * @param in pointer to input array
 */
struct nlop_s* nlop_set_input_const(const struct nlop_s* a, int i, int N, const long dims[N], bool copy, const complex float* in)
{
	return nlop_set_input_const2(a, i, N, dims, MD_STRIDES(N, dims, CFL_SIZE), copy, in);
}


/**
 * Chain operator with a constant operator and free the input operator
 * @param a operator whose input should be set constant
 * @param i index which should be set constant
 * @param N #dimensions of input array
 * @param dims dimensions of input array
 * @param strs strides of input array
 * @param copy decide if const in is copied in operator
 * @param in pointer to input array
 */
struct nlop_s* nlop_set_input_const_F2(const struct nlop_s* a, int i, int N, const long dims[N], const long strs[N], bool copy, const complex float* in)
{
	struct nlop_s* result = nlop_set_input_const2(a, i, N, dims, strs, copy, in);

	nlop_free(a);

	return result;
}


/**
 * Chain operator with a constant operator and free the input operator
 * @param a operator whose input should be set constant
 * @param i index which should be set constant
 * @param N #dimensions of input array
 * @param dims dimensions of input array
 * @param copy decide if const in is copied in operator
 * @param in pointer to input array
 */
struct nlop_s* nlop_set_input_const_F(const struct nlop_s* a, int i, int N, const long dims[N], bool copy, const complex float* in)
{
	struct nlop_s* result = nlop_set_input_const(a, i, N, dims, copy, in);

	nlop_free(a);

	return result;
}



struct del_out_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(del_out_s);

static void del_out_fun(const nlop_data_t* _data, int N, complex float** in)
{
	UNUSED(N);
	UNUSED(_data);
	UNUSED(in);
}

static void del_out_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(del_out_s, _data);

	xfree(data);
}


/**
 * Create operator with one input and zero outputs
 * @param N #dimensions
 * @param dims dimensions
 */
struct nlop_s* nlop_del_out_create(int N, const long dims[N])
{
	PTR_ALLOC(struct del_out_s, data);
	SET_TYPEID(del_out_s, data);

	long tdims[1][N];
	md_copy_dims(N, tdims[0], dims);

	return nlop_generic_create(0, 0, NULL, 1, N, tdims, CAST_UP(PTR_PASS(data)), del_out_fun, NULL, NULL, NULL,NULL, del_out_del);
}


/**
 * Returns a new operator without the output o
 * @param a operator
 * @param o index of output to be deleted
 */
struct nlop_s* nlop_del_out(const struct nlop_s* a, int o)
{
	int ao = nlop_get_nr_out_args(a);
	assert(ao > o);

	const struct iovec_s* codomain = nlop_generic_codomain(a, o);

	struct nlop_s* nlop_del_out_op = nlop_del_out_create(codomain->N, codomain->dims);
	struct nlop_s* result = nlop_chain2(a, o,  nlop_del_out_op,  0);

	nlop_free(nlop_del_out_op);

	return result;
}


/**
 * Returns a new operator without the output o
 * @param a operator
 * @param o index of output to be deleted
 */
struct nlop_s* nlop_del_out_F(const struct nlop_s* a, int o)
{
	auto result = nlop_del_out(a, o);

	nlop_free(a);

	return result;
}

