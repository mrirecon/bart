/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops_graph.h"

#include "nlops/nlop.h"
#include "num/ops.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "nlop_jacobian.h"


struct block_diag_s {

	INTERFACE(nlop_data_t);

	int II;
	int OO;

	int N;

	const struct iovec_s** iov_in;
	const struct iovec_s** iov_out;

	const struct iovec_s** iov_der;
	void** der;
	bool* holomorphic;
	void** derc;
	
	nlop_zrblock_diag_generic_fun_t zrblock_diag_fun;
	nlop_zblock_diag_generic_fun_t zblock_diag_fun;
	nlop_rblock_diag_generic_fun_t rblock_diag_fun;

	nlop_data_t* data;

	nlop_del_diag_fun_t del;
};

DEF_TYPEID(block_diag_s);

struct diag_s {

	INTERFACE(nlop_data_t);

	nlop_zrdiag_fun_t zrdiag_fun;
	nlop_zdiag_fun_t zdiag_fun;
	nlop_rdiag_fun_t rdiag_fun;

	nlop_del_diag_fun_t del;

	nlop_data_t* data;
};

DEF_TYPEID(diag_s);

struct block_diag_simple_s {

	INTERFACE(nlop_data_t);

	nlop_zrblock_diag_fun_t zrblock_diag_fun;
	nlop_zblock_diag_fun_t zblock_diag_fun;
	nlop_rblock_diag_fun_t rblock_diag_fun;

	nlop_del_diag_fun_t del;

	nlop_data_t* data;
};

DEF_TYPEID(block_diag_simple_s);

static void block_diag_clear_der(const nlop_data_t* _data)
{
	auto data = CAST_DOWN(block_diag_s, _data);

	int OO = data->OO;
	int II = data->II;

	void* (*der)[OO][II] = (void*)(data->der);
	void* (*derc)[OO][II] = (void*)(data->derc);

	for (int i = 0; i < II; i++) {
		for (int o = 0; o < OO; o++) {

			md_free((*der)[o][i]);
			(*der)[o][i] = NULL;

			md_free((*derc)[o][i]);
			(*derc)[o][i] = NULL;
		}
	}
}

static void zrblock_diag_fun(const nlop_data_t* _data, int Nargs, complex float* args[Nargs])
{
	const auto data = CAST_DOWN(block_diag_s, _data);

	int OO = data->OO;
	int II = data->II;

	int N = data->N;

	assert(data->OO + data->II == Nargs);

	long idims[II][N];
	const complex float* src[II];

	for (int i = 0; i < II; i++) {

		md_copy_dims(N, idims[i], data->iov_in[i]->dims);
		src[i] = args[OO + i];
	}


	long odims[OO][N];
	complex float* dst[OO];

	for (int i = 0; i < OO; i++) {

		md_copy_dims(N, odims[i], data->iov_out[i]->dims);
		dst[i] = args[i];
	}


	long ddims[OO][II][N];
	complex float* (*der)[OO][II] = (void*)(data->der);
	complex float* (*derc)[OO][II] = (void*)(data->derc);
	bool (*holom)[OO][II] = (void*)(data->holomorphic);
	const struct iovec_s* (*iov_der)[OO][II] = (void*)(data->iov_der);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {

			auto iov = (*iov_der)[o][i];

			md_copy_dims(N, ddims[o][i], iov->dims);

			md_free((*der)[o][i]);
			(*der)[o][i] = NULL;

			if (nlop_der_requested(_data, i, o))
				(*der)[o][i] = md_alloc_sameplace(iov->N, iov->dims, iov->size, args[0]);

			md_free((*derc)[o][i]);
			(*derc)[o][i] = NULL;

			if (!(*holom)[o][i] && nlop_der_requested(_data, i, o))
				(*derc)[o][i] = md_alloc_sameplace(iov->N, iov->dims, iov->size, args[0]);
		}

	assert(NULL == data->rblock_diag_fun);

	if (NULL != data->zblock_diag_fun)
		data->zblock_diag_fun(data->data, N, OO, odims, dst, II, idims, src, ddims, (*der));
	
	if (NULL != data->zrblock_diag_fun)
		data->zrblock_diag_fun(data->data, N, OO, odims, dst, II, idims, src, ddims, (*der), (*derc));
}

static void zblock_diag_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(block_diag_s, _data);

	int OO = data->OO;
	int II = data->II;
	const complex float* der = (*(complex float* (*)[OO][II])(data->der))[o][i];
	const long* ddims = (*(const struct iovec_s* (*)[OO][II])(data->iov_der))[o][i]->dims;

	if (NULL == der)
		error("Block diag %x derivative not available!\n", data);

	md_ztenmul(data->N, data->iov_out[o]->dims, dst, data->iov_in[i]->dims, src, ddims, der);

	const complex float* derc = (*(complex float* (*)[OO][II])(data->derc))[o][i];

	long max_dims[data->N];
	md_tenmul_dims(data->N, max_dims, data->iov_out[o]->dims, data->iov_in[i]->dims, ddims);

	if (NULL != derc)
		md_zfmacc2(data->N, max_dims,
			data->iov_out[o]->strs, dst,
			MD_STRIDES(data->N, ddims, CFL_SIZE), derc,
			data->iov_in[i]->strs, src
			);
}

static void zblock_diag_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(block_diag_s, _data);

	int OO = data->OO;
	int II = data->II;
	const complex float* der = (*(complex float* (*)[OO][II])(data->der))[o][i];
	const long* ddims = (*(const struct iovec_s* (*)[OO][II])(data->iov_der))[o][i]->dims;

	if (NULL == der)
		error("Block diag %x derivative not available!\n", data);

	md_ztenmulc(data->N, data->iov_in[i]->dims, dst, data->iov_out[o]->dims, src, ddims, der);

	const complex float* derc = (*(complex float* (*)[OO][II])(data->derc))[o][i];

	long max_dims[data->N];
	md_tenmul_dims(data->N, max_dims, data->iov_out[o]->dims, data->iov_in[i]->dims, ddims);

	if (NULL != derc)
		md_zfmacc2(data->N, max_dims,
			data->iov_in[i]->strs, dst,
			MD_STRIDES(data->N, ddims, CFL_SIZE), derc,
			data->iov_out[o]->strs, src);
}

static void block_diag_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(block_diag_s, _data);

	if (NULL != data->del)
		data->del(data->data);
	else if (NULL != data->data)
		xfree(data->data);

	for (int i = 0; i < data->II; i++)
		iovec_free(data->iov_in[i]);

	for (int i = 0; i < data->OO; i++)
		iovec_free(data->iov_out[i]);

	for (int i = 0; i < data->OO * data->II; i++) {

		iovec_free(data->iov_der[i]);
		md_free(data->der[i]);
		md_free(data->derc[i]);
	}

	xfree(data->iov_in);
	xfree(data->iov_out);
	xfree(data->iov_der);
	xfree(data->der);
	xfree(data->derc);

	xfree(data->holomorphic);

	xfree(data);
}

static const struct graph_s* nlop_block_diag_get_graph(const struct operator_s* op, const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(block_diag_s, _data);
	if (NULL != data)
		_data = data->data;

	if (NULL != CAST_MAYBE(diag_s, _data)) {

		const auto data = CAST_DOWN(diag_s, _data);
		if (NULL != data)
			_data = data->data;
	}

	if (NULL != CAST_MAYBE(block_diag_simple_s, _data)) {

		const auto data = CAST_DOWN(block_diag_simple_s, _data);
		if (NULL != data)
			_data = data->data;
	}

	return create_graph_operator(op, _data->TYPEID->name);
}

struct nlop_s* nlop_zrblock_diag_generic_create(nlop_data_t* data, int N,
						int OO, const long odims[OO][N],
						int II, const long idims[II][N],
						unsigned long diag_flags [OO][II],
						bool holomorphic[OO][II],
						nlop_zrblock_diag_generic_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct block_diag_s, _data);
	SET_TYPEID(block_diag_s, _data);

	_data->iov_in = *TYPE_ALLOC(const struct iovec_s*[II]);
	_data->iov_out = *TYPE_ALLOC(const struct iovec_s*[OO]);

	_data->iov_der = &((*TYPE_ALLOC(const struct iovec_s*[OO][II]))[0][0]);
	_data->der = &((*TYPE_ALLOC(void*[OO][II]))[0][0]);
	_data->derc = &((*TYPE_ALLOC(void*[OO][II]))[0][0]);

	_data->holomorphic = &(ARR_CLONE(bool[OO][II], holomorphic)[0][0]);

	_data->data = data;
	_data->del = del;

	_data->rblock_diag_fun = NULL;
	_data->zrblock_diag_fun = forward;
	_data->zblock_diag_fun = NULL;

	_data->N = N;
	_data->OO = OO;
	_data->II = II;

	nlop_der_fun_t der_funs[II][OO];
	nlop_der_fun_t adj_funs[II][OO];

	for (int i = 0; i < II; i++)
		_data->iov_in[i] = iovec_create(N, idims[i], CFL_SIZE);

	for (int i = 0; i < OO; i++)
		_data->iov_out[i] = iovec_create(N, odims[i], CFL_SIZE);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {

			der_funs[i][o] = zblock_diag_der;
			adj_funs[i][o] = zblock_diag_adj;

			assert(md_check_compat(N, ~0, odims[o], idims[i]));

			long ddims[N];
			md_singleton_dims(N, ddims);
			md_max_dims(N, ~diag_flags[o][i], ddims, odims[o], idims[i]);

			(*(const struct iovec_s* (*)[OO][II])(_data->iov_der))[o][i] = iovec_create(N, ddims, CFL_SIZE);
			(*(void* (*)[OO][II])(_data->der))[o][i] = NULL;
			(*(void* (*)[OO][II])(_data->derc))[o][i] = NULL;
		}

	return nlop_generic_managed_create(
		OO, N, odims, II, N, idims, CAST_UP(PTR_PASS(_data)),
		zrblock_diag_fun, der_funs, adj_funs,
		NULL, NULL, block_diag_del, block_diag_clear_der, nlop_block_diag_get_graph
	);
}


struct nlop_s* nlop_zblock_diag_generic_create(nlop_data_t* data, int N,
						int OO, const long odims[OO][N],
						int II, const long idims[II][N],
						unsigned long diag_flags [OO][II],
						nlop_zblock_diag_generic_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct block_diag_s, _data);
	SET_TYPEID(block_diag_s, _data);

	_data->iov_in = *TYPE_ALLOC(const struct iovec_s*[II]);
	_data->iov_out = *TYPE_ALLOC(const struct iovec_s*[OO]);

	_data->iov_der = &((*TYPE_ALLOC(const struct iovec_s*[OO][II]))[0][0]);
	_data->der = &((*TYPE_ALLOC(void*[OO][II]))[0][0]);
	_data->derc = &((*TYPE_ALLOC(void*[OO][II]))[0][0]);

	bool* holom = &((*TYPE_ALLOC(bool[OO][II]))[0][0]);
	for (int i = 0; i < (II * OO); i++)
		holom[i] = true;
	_data->holomorphic = holom;

	_data->data = data;
	_data->del = del;

	_data->rblock_diag_fun = NULL;
	_data->zrblock_diag_fun = NULL;
	_data->zblock_diag_fun = forward;

	_data->N = N;
	_data->OO = OO;
	_data->II = II;

	nlop_der_fun_t der_funs[II][OO];
	nlop_der_fun_t adj_funs[II][OO];

	for (int i = 0; i < II; i++)
		_data->iov_in[i] = iovec_create(N, idims[i], CFL_SIZE);

	for (int i = 0; i < OO; i++)
		_data->iov_out[i] = iovec_create(N, odims[i], CFL_SIZE);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {

			der_funs[i][o] = zblock_diag_der;
			adj_funs[i][o] = zblock_diag_adj;

			assert(md_check_compat(N, ~0, odims[o], idims[i]));

			long ddims[N];
			md_singleton_dims(N, ddims);
			md_max_dims(N, ~diag_flags[o][i], ddims, odims[o], idims[i]);

			(*(const struct iovec_s* (*)[OO][II])(_data->iov_der))[o][i] = iovec_create(N, ddims, CFL_SIZE);
			(*(void* (*)[OO][II])(_data->der))[o][i] = NULL;
			(*(void* (*)[OO][II])(_data->derc))[o][i] = NULL;
		}

	return nlop_generic_managed_create(
		OO, N, odims, II, N, idims, CAST_UP(PTR_PASS(_data)),
		zrblock_diag_fun, der_funs, adj_funs,
		NULL, NULL, block_diag_del, block_diag_clear_der, nlop_block_diag_get_graph
	);
}



static void rblock_diag_fun(const nlop_data_t* _data, int Nargs, complex float* args[Nargs])
{
	const auto data = CAST_DOWN(block_diag_s, _data);

	int OO = data->OO;
	int II = data->II;

	int N = data->N;

	assert(data->OO + data->II == Nargs);

	long idims[II][N];
	const float* src[II];

	for (int i = 0; i < II; i++) {

		md_copy_dims(N, idims[i], data->iov_in[i]->dims);
		src[i] = (float*)args[OO + i];
	}


	long odims[OO][N];
	float* dst[OO];

	for (int i = 0; i < OO; i++) {

		md_copy_dims(N, odims[i], data->iov_out[i]->dims);
		dst[i] = (float*)args[i];
	}


	long ddims[OO][II][N];
	float* (*der)[OO][II] = (void*)(data->der);
	const struct iovec_s* (*iov_der)[OO][II] = (void*)(data->iov_der);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {

			auto iov = (*iov_der)[o][i];

			md_copy_dims(N, ddims[o][i], iov->dims);

			md_free((*der)[o][i]);
			(*der)[o][i] = NULL;

			if (nlop_der_requested(_data, i, o))
				(*der)[o][i] = md_alloc_sameplace(iov->N, iov->dims, iov->size, args[0]);
		}

	assert(NULL == data->zblock_diag_fun);

	data->rblock_diag_fun(data->data, N, OO, odims, dst, II, idims, src, ddims, (*der));
}

static void rblock_diag_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(block_diag_s, _data);
	int OO = data->OO;
	int II = data->II;
	const float* der = (*(float* (*)[OO][II])(data->der))[o][i];
	const long* ddims = (*(const struct iovec_s* (*)[OO][II])(data->iov_der))[o][i]->dims;

	if (NULL == der)
		error("Block diag %x derivative not available!\n", data);

	md_tenmul(data->N, data->iov_out[o]->dims, (float*)dst, data->iov_in[i]->dims, (float*)src, ddims, der);
}

static void rblock_diag_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(block_diag_s, _data);

	int OO = data->OO;
	int II = data->II;
	const float* der = (*(float* (*)[OO][II])(data->der))[o][i];
	const long* ddims = (*(const struct iovec_s* (*)[OO][II])(data->iov_der))[o][i]->dims;

	if (NULL == der)
		error("Block diag %x derivative not available!\n", data);

	md_tenmul(data->N, data->iov_in[i]->dims, (float*)dst, data->iov_out[o]->dims, (float*)src, ddims, der);
}

struct nlop_s* nlop_rblock_diag_generic_create(nlop_data_t* data, int N,
						int OO, const long rodims[OO][N],
						int II, const long ridims[II][N],
						unsigned long diag_flags [OO][II],
						nlop_rblock_diag_generic_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct block_diag_s, _data);
	SET_TYPEID(block_diag_s, _data);

	_data->iov_in = *TYPE_ALLOC(const struct iovec_s*[II]);
	_data->iov_out = *TYPE_ALLOC(const struct iovec_s*[OO]);

	_data->iov_der = &((*TYPE_ALLOC(const struct iovec_s*[OO][II]))[0][0]);
	_data->der = &((*TYPE_ALLOC(void*[OO][II]))[0][0]);
	_data->derc = &((*TYPE_ALLOC(void*[OO][II]))[0][0]);

	bool* holom = &((*TYPE_ALLOC(bool[OO][II]))[0][0]);
	for (int i = 0; i < (II * OO); i++)
		holom[i] = true;
	_data->holomorphic = holom;

	_data->data = data;
	_data->del = del;

	_data->zblock_diag_fun = NULL;
	_data->zrblock_diag_fun = NULL;
	_data->rblock_diag_fun = forward;

	assert(2 < N);

	_data->N = N;
	_data->OO = OO;
	_data->II = II;

	long odims[OO][N - 2];
	long idims[II][N - 2];

	nlop_der_fun_t der_funs[II][OO];
	nlop_der_fun_t adj_funs[II][OO];

	for (int i = 0; i < II; i++) {

		assert(2 == ridims[i][0] * ridims[i][1]);

		md_copy_dims(N - 2, idims[i], ridims[i] + 2);
		_data->iov_in[i] = iovec_create(N, ridims[i], FL_SIZE);
	}

	for (int i = 0; i < OO; i++){

		assert(2 == rodims[i][0] * rodims[i][1]);

		md_copy_dims(N - 2, odims[i], rodims[i] + 2);
		_data->iov_out[i] = iovec_create(N, rodims[i], FL_SIZE);
	}

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {

			der_funs[i][o] = rblock_diag_der;
			adj_funs[i][o] = rblock_diag_adj;

			assert(md_check_compat(N, ~0, rodims[o], ridims[i]));

			long ddims[N];
			md_singleton_dims(N, ddims);
			md_max_dims(N, ~diag_flags[o][i], ddims, rodims[o], ridims[i]);

			(*(const struct iovec_s* (*)[OO][II])(_data->iov_der))[o][i] = iovec_create(N, ddims, FL_SIZE);
			(*(void* (*)[OO][II])(_data->der))[o][i] = NULL;
			(*(void* (*)[OO][II])(_data->derc))[o][i] = NULL;
		}

	return nlop_generic_managed_create(
		OO, N - 2, odims, II, N - 2, idims, CAST_UP(PTR_PASS(_data)),
		rblock_diag_fun, der_funs, adj_funs,
		NULL, NULL, block_diag_del, block_diag_clear_der, nlop_block_diag_get_graph
	);
}


bool nlop_block_diag_der_available(const struct nlop_s* op, int o, int i)
{
	auto data = CAST_MAYBE(block_diag_s, nlop_get_data((struct nlop_s*)op));
	assert(NULL != data);

	return (NULL != (*(void* (*)[data->OO][data->II])(data->der))[o][i]);
}


static void zrdiag_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], _Complex float* dst[OO], int II, const long idims[II][N], const _Complex float* src[II], const long ddims[OO][II][N], _Complex float* jac[OO][II], _Complex float* jacc[OO][II])
{
	auto data = CAST_DOWN(diag_s, _data);

	assert(1 == OO);
	assert(1 == II);

	assert(md_check_equal_dims(N, idims[0], ddims[0][0], ~0));
	assert(md_check_equal_dims(N, odims[0], ddims[0][0], ~0));

	assert(NULL == data->rdiag_fun);
	assert(NULL == data->zdiag_fun);
	data->zrdiag_fun(data->data, N, odims[0], dst[0], src[0], jac[0][0], jacc[0][0]);
}


static void diag_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(diag_s, _data);

	if (NULL != data->del)
		data->del(data->data);
	else if (NULL != data->data)
		xfree(data->data);

	xfree(data);
}


struct nlop_s* nlop_zrdiag_create(int N, const long dims[N], nlop_data_t* data, nlop_zrdiag_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct diag_s, _data);
	SET_TYPEID(diag_s, _data);

	_data->data = data;
	_data->del = del;
	_data->rdiag_fun = NULL;
	_data->zdiag_fun = NULL;
	_data->zrdiag_fun = forward;

	long nl_odims[1][N];
	long nl_idims[1][N];

	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_odims[0], dims);

	unsigned long diag_flags[1][1];
	diag_flags[0][0] = 0;
	
	bool holomorphic[1][1];
	holomorphic[0][0] = false;

	return nlop_zrblock_diag_generic_create(CAST_UP(PTR_PASS(_data)), N, 1, nl_odims, 1, nl_idims, diag_flags, holomorphic, zrdiag_fun, diag_del);
}




static void zdiag_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], _Complex float* dst[OO], int II, const long idims[II][N], const _Complex float* src[II], const long ddims[OO][II][N], _Complex float* jac[OO][II])
{
	auto data = CAST_DOWN(diag_s, _data);

	assert(1 == OO);
	assert(1 == II);

	assert(md_check_equal_dims(N, idims[0], ddims[0][0], ~0));
	assert(md_check_equal_dims(N, odims[0], ddims[0][0], ~0));

	assert(NULL == data->rdiag_fun);
	assert(NULL == data->zrdiag_fun);
	data->zdiag_fun(data->data, N, odims[0], dst[0], src[0], jac[0][0]);
}



struct nlop_s* nlop_zdiag_create(int N, const long dims[N], nlop_data_t* data, nlop_zdiag_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct diag_s, _data);
	SET_TYPEID(diag_s, _data);

	_data->data = data;
	_data->del = del;
	_data->rdiag_fun = NULL;
	_data->zdiag_fun = forward;
	_data->zrdiag_fun = NULL;

	long nl_odims[1][N];
	long nl_idims[1][N];

	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_odims[0], dims);

	unsigned long diag_flags[1][1];
	diag_flags[0][0] = 0;

	return nlop_zblock_diag_generic_create(CAST_UP(PTR_PASS(_data)), N, 1, nl_odims, 1, nl_idims, diag_flags, zdiag_fun, diag_del);
}


static void rdiag_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], float* dst[OO], int II, const long idims[II][N], const float* src[II], const long ddims[OO][II][N], float* jac[OO][II])
{
	auto data = CAST_DOWN(diag_s, _data);

	assert(1 == OO);
	assert(1 == II);

	assert(md_check_equal_dims(N, idims[0], ddims[0][0], ~0));
	assert(md_check_equal_dims(N, odims[0], ddims[0][0], ~0));

	assert(NULL == data->zdiag_fun);
	data->rdiag_fun(data->data, N, odims[0], (float*)(dst[0]), (float*)(src[0]), (float*)(jac[0][0]));
}

struct nlop_s* nlop_rdiag_create(int N, const long dims[N], nlop_data_t* data, nlop_rdiag_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct diag_s, _data);
	SET_TYPEID(diag_s, _data);

	_data->data = data;
	_data->del = del;
	_data->rdiag_fun = forward;
	_data->zdiag_fun = NULL;
	_data->zrdiag_fun = NULL;

	long nl_odims[1][N + 2];
	long nl_idims[1][N + 2];

	md_copy_dims(N, nl_idims[0] + 2, dims);
	md_copy_dims(N, nl_odims[0] + 2, dims);

	nl_odims[0][0] = 1;
	nl_idims[0][0] = 1;

	nl_odims[0][1] = 2;
	nl_idims[0][1] = 2;

	unsigned long diag_flags[1][1];
	diag_flags[0][0] = 0;

	return nlop_rblock_diag_generic_create(CAST_UP(PTR_PASS(_data)), N + 2, 1, nl_odims, 1, nl_idims, diag_flags, rdiag_fun, diag_del);
}





static void zblock_diag_simple_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], _Complex float* dst[OO], int II, const long idims[II][N], const _Complex float* src[II], const long ddims[OO][II][N], _Complex float* jac[OO][II])
{
	auto data = CAST_DOWN(block_diag_simple_s, _data);

	assert(1 == OO);
	assert(1 == II);

	assert(NULL == data->rblock_diag_fun);
	assert(NULL == data->zrblock_diag_fun);
	data->zblock_diag_fun(data->data, N, odims[0], dst[0], idims[0], src[0], ddims[0][0], jac[0][0]);
}


static void block_diag_simple_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(block_diag_simple_s, _data);

	if (NULL != data->del)
		data->del(data->data);
	else if (NULL != data->data)
		xfree(data->data);

	xfree(data);
}

struct nlop_s* nlop_zblock_diag_create(nlop_data_t* data, int N, const long odims[N], const long idims[N], const long ddims[N], nlop_zblock_diag_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct block_diag_simple_s, _data);
	SET_TYPEID(block_diag_simple_s, _data);

	_data->data = data;
	_data->del = del;
	_data->rblock_diag_fun = NULL;
	_data->zblock_diag_fun = forward;
	_data->zrblock_diag_fun = NULL;

	long nl_odims[1][N];
	long nl_idims[1][N];

	md_copy_dims(N, nl_idims[0], idims);
	md_copy_dims(N, nl_odims[0], odims);

	unsigned long diag_flags[1][1];
	diag_flags[0][0] = ~md_nontriv_dims(N, ddims);

	return nlop_zblock_diag_generic_create(CAST_UP(PTR_PASS(_data)), N, 1, nl_odims, 1, nl_idims, diag_flags, zblock_diag_simple_fun, block_diag_simple_del);
}

static void rblock_diag_simple_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], float* dst[OO], int II, const long idims[II][N], const float* src[II], const long ddims[OO][II][N], float* jac[OO][II])
{
	auto data = CAST_DOWN(block_diag_simple_s, _data);

	assert(1 == OO);
	assert(1 == II);

	assert(NULL == data->zblock_diag_fun);
	data->rblock_diag_fun(data->data, N, odims[0], dst[0], idims[0], src[0], ddims[0][0], jac[0][0]);
}

static void zrblock_diag_simple_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], _Complex float* dst[OO], int II, const long idims[II][N], const _Complex float* src[II], const long ddims[OO][II][N], _Complex float* jac[OO][II], _Complex float* jacc[OO][II])
{
	auto data = CAST_DOWN(block_diag_simple_s, _data);

	assert(1 == OO);
	assert(1 == II);

	assert(NULL == data->rblock_diag_fun);
	assert(NULL == data->zblock_diag_fun);
	data->zrblock_diag_fun(data->data, N, odims[0], dst[0], idims[0], src[0], ddims[0][0], jac[0][0], jacc[0][0]);
}

struct nlop_s* nlop_zrblock_diag_create(nlop_data_t* data, int N, const long odims[N], const long idims[N], const long ddims[N], nlop_zrblock_diag_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct block_diag_simple_s, _data);
	SET_TYPEID(block_diag_simple_s, _data);

	_data->data = data;
	_data->del = del;
	_data->rblock_diag_fun = NULL;
	_data->zblock_diag_fun = NULL;
	_data->zrblock_diag_fun = forward;

	long nl_odims[1][N];
	long nl_idims[1][N];

	md_copy_dims(N, nl_idims[0], idims);
	md_copy_dims(N, nl_odims[0], odims);

	unsigned long diag_flags[1][1];
	diag_flags[0][0] = ~md_nontriv_dims(N, ddims);

	bool holomorphic[1][1];
	holomorphic[0][0] = false;

	return nlop_zrblock_diag_generic_create(CAST_UP(PTR_PASS(_data)), N, 1, nl_odims, 1, nl_idims, diag_flags, holomorphic, zrblock_diag_simple_fun, block_diag_simple_del);
}


struct nlop_s* nlop_rblock_diag_create(nlop_data_t* data, int N, const long odims[N], const long idims[N], const long ddims[N], nlop_rblock_diag_fun_t forward, nlop_del_diag_fun_t del)
{
	PTR_ALLOC(struct block_diag_simple_s, _data);
	SET_TYPEID(block_diag_simple_s, _data);

	_data->data = data;
	_data->del = del;
	_data->rblock_diag_fun = forward;
	_data->zblock_diag_fun = NULL;

	long nl_odims[1][N];
	long nl_idims[1][N];

	md_copy_dims(N, nl_idims[0], idims);
	md_copy_dims(N, nl_odims[0], odims);

	unsigned long diag_flags[1][1];
	diag_flags[0][0] = ~md_nontriv_dims(N, ddims);

	return nlop_rblock_diag_generic_create(CAST_UP(PTR_PASS(_data)), N, 1, nl_odims, 1, nl_idims, diag_flags, rblock_diag_simple_fun, block_diag_simple_del);
}

void linop_compute_matrix_zblock_diag_fwd(const struct linop_s* lop, int N, const long ddims[N], complex float* jacobian)
{
	assert(N == (int)linop_domain(lop)->N);
	assert(N == (int)linop_codomain(lop)->N);

	const long* odims = linop_codomain(lop)->dims;
	const long* idims = linop_domain(lop)->dims;

	long ostrs[N];
	long istrs[N];
	long dstrs[N];

	md_calc_strides(N, ostrs, odims, CFL_SIZE);
	md_calc_strides(N, istrs, idims, CFL_SIZE);
	md_calc_strides(N, dstrs, ddims, CFL_SIZE);

	long mdims[N];
	md_max_dims(N, ~0, mdims, odims, idims);
	assert(md_check_equal_dims(N, mdims, ddims, ~0));
	assert(md_check_compat(N, ~0, odims, ddims));
	assert(md_check_compat(N, ~0, idims, ddims));

	unsigned long loop_flags = md_nontriv_dims(N, idims) & ~md_nontriv_dims(N, odims);
	long diag_dims[N];
	md_select_dims(N, ~loop_flags, diag_dims, idims);

	long pos[N];
	md_singleton_strides(N, pos);

	complex float* in = md_alloc_sameplace(N, idims, CFL_SIZE, jacobian);
	complex float* out = md_alloc_sameplace(N, odims, CFL_SIZE, jacobian);
	complex float* ones = md_alloc_sameplace(N, diag_dims, CFL_SIZE, jacobian);
	md_zfill(N, diag_dims, ones, 1);

	do {

		md_clear(N, idims, in, CFL_SIZE);
		md_copy_block(N, pos, idims, in, diag_dims, ones, CFL_SIZE);
		linop_forward_unchecked(lop, out, in);
		md_copy_block(N, pos, ddims, jacobian, odims, out, CFL_SIZE);

	} while (md_next(N, idims, loop_flags, pos));

	md_free(in);
	md_free(out);
	md_free(ones);
}

void linop_compute_matrix_zblock_diag_bwd(const struct linop_s* lop, int N, const long ddims[N], complex float* jacobian)
{
	assert(N == (int)linop_domain(lop)->N);
	assert(N == (int)linop_codomain(lop)->N);

	const long* odims = linop_codomain(lop)->dims;
	const long* idims = linop_domain(lop)->dims;

	long ostrs[N];
	long istrs[N];
	long dstrs[N];

	md_calc_strides(N, ostrs, odims, CFL_SIZE);
	md_calc_strides(N, istrs, idims, CFL_SIZE);
	md_calc_strides(N, dstrs, ddims, CFL_SIZE);

	long mdims[N];
	md_max_dims(N, ~0, mdims, odims, idims);
	assert(md_check_equal_dims(N, mdims, ddims, ~0));
	assert(md_check_compat(N, ~0, odims, ddims));
	assert(md_check_compat(N, ~0, idims, ddims));

	unsigned long loop_flags = ~md_nontriv_dims(N, idims) & md_nontriv_dims(N, odims);
	long diag_dims[N];
	md_select_dims(N, ~loop_flags, diag_dims, odims);

	long pos[N];
	md_singleton_strides(N, pos);

	complex float* in = md_alloc_sameplace(N, idims, CFL_SIZE, jacobian);
	complex float* out = md_alloc_sameplace(N, odims, CFL_SIZE, jacobian);
	complex float* ones = md_alloc_sameplace(N, diag_dims, CFL_SIZE, jacobian);
	md_zfill(N, diag_dims, ones, 1);

	do {

		md_clear(N, odims, out, CFL_SIZE);
		md_copy_block(N, pos, odims, out, diag_dims, ones, CFL_SIZE);
		linop_adjoint_unchecked(lop, in, out);
		md_copy_block(N, pos, ddims, jacobian, idims, in, CFL_SIZE);

	} while (md_next(N, odims, loop_flags, pos));

	md_zconj(N, ddims, jacobian, jacobian);

	md_free(in);
	md_free(out);
	md_free(ones);
}

void linop_compute_matrix_zblock_diag(const struct linop_s* lop, int N, const long ddims[N], complex float* jacobian)
{
	assert(N == (int)linop_domain(lop)->N);
	assert(N == (int)linop_codomain(lop)->N);

	const long* odims = linop_codomain(lop)->dims;
	const long* idims = linop_domain(lop)->dims;

	long odims2[N];
	long idims2[N];

	md_select_dims(N, ~md_nontriv_dims(N, odims), idims2, idims);
	md_select_dims(N, ~md_nontriv_dims(N, idims), odims2, odims);

	if (md_calc_size(N, odims2) > md_calc_size(N, idims2))
		linop_compute_matrix_zblock_diag_fwd(lop, N, ddims, jacobian);
	else
		linop_compute_matrix_zblock_diag_bwd(lop, N, ddims, jacobian);
}

void linop_compute_matrix_rblock_diag_fwd(const struct linop_s* lop, int N, const long ddims[N], float* jacobian)
{
	assert(N == 2 + (int)linop_domain(lop)->N);
	assert(N == 2 + (int)linop_codomain(lop)->N);

	long odims[N];
	long idims[N];

	odims[0] = 2;
	odims[1] = 1;

	idims[0] = 1;
	idims[1] = 2;

	assert(2 == ddims[0]);
	assert(2 == ddims[1]);

	md_copy_dims(N - 2, odims + 2, linop_codomain(lop)->dims);
	md_copy_dims(N - 2, idims + 2, linop_domain(lop)->dims);

	long ostrs[N];
	long istrs[N];
	long dstrs[N];

	md_calc_strides(N, ostrs, odims, FL_SIZE);
	md_calc_strides(N, istrs, idims, FL_SIZE);
	md_calc_strides(N, dstrs, ddims, FL_SIZE);

	long mdims[N];
	md_max_dims(N, ~0, mdims, odims, idims);
	assert(md_check_equal_dims(N, mdims, ddims, ~0));
	assert(md_check_compat(N, ~0, odims, ddims));
	assert(md_check_compat(N, ~0, idims, ddims));

	unsigned long loop_flags = md_nontriv_dims(N, idims) & ~md_nontriv_dims(N, odims);
	long diag_dims[N];
	md_select_dims(N, ~loop_flags, diag_dims, idims);

	long pos[N];
	md_singleton_strides(N, pos);

	float* in = md_alloc_sameplace(N, idims, FL_SIZE, jacobian);
	float* out = md_alloc_sameplace(N, odims, FL_SIZE, jacobian);

	complex float* ones = md_alloc_sameplace(N, diag_dims, CFL_SIZE, jacobian);
	complex float* imag = md_alloc_sameplace(N, diag_dims, CFL_SIZE, jacobian);

	md_zfill(N, diag_dims, ones, 1.);
	md_zfill(N, diag_dims, imag, 1. * I);

	do {

		md_clear(N, idims, in, FL_SIZE);

		md_copy_block(N - 2, pos + 2, idims + 2, in, diag_dims + 2, ones, CFL_SIZE);
		linop_forward_unchecked(lop, (complex float*)out, (complex float*)in);
		md_copy_block(N, pos, ddims, jacobian, odims, out, FL_SIZE);

		pos[1] = 1;

		md_copy_block(N - 2, pos + 2, idims + 2, in, diag_dims + 2, imag, CFL_SIZE);
		linop_forward_unchecked(lop, (complex float*)out, (complex float*)in);
		md_copy_block(N, pos, ddims, jacobian, odims, out, FL_SIZE);

	} while (md_next(N, idims, loop_flags, pos));

	md_free(in);
	md_free(out);
	md_free(ones);
	md_free(imag);
}

void linop_compute_matrix_rblock_diag_bwd(const struct linop_s* lop, int N, const long ddims[N], float* jacobian)
{
	assert(N == 2 + (int)linop_domain(lop)->N);
	assert(N == 2 + (int)linop_codomain(lop)->N);

	long odims[N];
	long idims[N];

	odims[0] = 2;
	odims[1] = 1;

	idims[0] = 1;
	idims[1] = 2;

	md_copy_dims(N - 2, odims + 2, linop_codomain(lop)->dims);
	md_copy_dims(N - 2, idims + 2, linop_domain(lop)->dims);

	long ostrs[N];
	long istrs[N];
	long dstrs[N];

	md_calc_strides(N, ostrs, odims, FL_SIZE);
	md_calc_strides(N, istrs, idims, FL_SIZE);
	md_calc_strides(N, dstrs, ddims, FL_SIZE);

	long mdims[N];
	md_max_dims(N, ~0, mdims, odims, idims);
	assert(md_check_equal_dims(N, mdims, ddims, ~0));
	assert(md_check_compat(N, ~0, odims, ddims));
	assert(md_check_compat(N, ~0, idims, ddims));

	unsigned long loop_flags = ~md_nontriv_dims(N, idims) & md_nontriv_dims(N, odims);
	long diag_dims[N];
	md_select_dims(N, ~loop_flags, diag_dims, odims);

	long pos[N];
	md_singleton_strides(N, pos);

	float* in = md_alloc_sameplace(N, idims, FL_SIZE, jacobian);
	float* out = md_alloc_sameplace(N, odims, FL_SIZE, jacobian);
	
	complex float* ones = md_alloc_sameplace(N, diag_dims, CFL_SIZE, jacobian);
	complex float* imag = md_alloc_sameplace(N, diag_dims, CFL_SIZE, jacobian);

	md_zfill(N, diag_dims, ones, 1.);
	md_zfill(N, diag_dims, imag, 1. * I);

	do {

		md_clear(N, odims, out, FL_SIZE);

		md_copy_block(N - 2, pos + 2, odims + 2, out, diag_dims + 2, ones, CFL_SIZE);
		linop_adjoint_unchecked(lop, (complex float*)in, (complex float*)out);
		md_copy_block(N, pos, ddims, jacobian, idims, in, FL_SIZE);

		pos[0] = 1;

		md_copy_block(N - 2, pos + 2, odims + 2, out, diag_dims + 2, imag, CFL_SIZE);
		linop_adjoint_unchecked(lop, (complex float*)in, (complex float*)out);
		md_copy_block(N, pos, ddims, jacobian, idims, in, FL_SIZE);

	} while (md_next(N, odims, loop_flags, pos));

	md_free(in);
	md_free(out);
	md_free(ones);
	md_free(imag);
}

void linop_compute_matrix_rblock_diag(const struct linop_s* lop, int N, const long ddims[N], float* jacobian)
{
	assert(N == 2 + (int)linop_domain(lop)->N);
	assert(N == 2 + (int)linop_codomain(lop)->N);

	const long* odims = linop_codomain(lop)->dims;
	const long* idims = linop_domain(lop)->dims;

	long odims2[N - 2];
	long idims2[N - 2];

	md_select_dims(N - 2, ~md_nontriv_dims(N - 2, odims), idims2, idims);
	md_select_dims(N - 2, ~md_nontriv_dims(N - 2, idims), odims2, odims);

	if (md_calc_size(N - 2, odims2) > md_calc_size(N - 2, idims2))
		linop_compute_matrix_rblock_diag_fwd(lop, N, ddims, jacobian);
	else
		linop_compute_matrix_rblock_diag_bwd(lop, N, ddims, jacobian);
}


void linop_compute_matrix_zrblock_diag(const struct linop_s* lop, int N, const long ddims[N], complex float* jac, complex float* jacc)
{
	assert(N == (int)linop_domain(lop)->N);
	assert(N == (int)linop_codomain(lop)->N);

	long ddims2[N + 2];
	ddims2[0] = 2;
	ddims2[1] = 2;
	md_copy_dims(N, ddims2 + 2, ddims);

	float* jac_r = md_alloc_sameplace(N + 2, ddims2, FL_SIZE, jac);

	linop_compute_matrix_rblock_diag(lop, N + 2, ddims2, jac_r);

	complex float* jac_y = md_alloc_sameplace(N, ddims, CFL_SIZE, jac);
	complex float* jac_x = md_alloc_sameplace(N, ddims, CFL_SIZE, jac);

	long pos[N + 2];
	for (int i = 0; i < N + 2; i++)
		pos[i] = 0;
	
	int idx = 1;
	
	md_slice(N + 2, MD_BIT(idx), pos, ddims2, jac_x, jac_r, FL_SIZE);
	
	pos[idx] = 1;
	md_slice(N + 2, MD_BIT(idx), pos, ddims2, jac_y, jac_r, FL_SIZE);

	md_zsmul(N, ddims, jac,  jac_x, 0.5);
	md_zsmul(N, ddims, jacc, jac_x, 0.5);

	md_zaxpy(N, ddims, jac,  -0.5 * I, jac_y);
	md_zaxpy(N, ddims, jacc, +0.5 * I, jac_y);

	md_free(jac_r);
	md_free(jac_x);
	md_free(jac_y);
}



struct precomp_jacobian_s {

	INTERFACE(nlop_data_t);

	const struct nlop_s* nlop;
};

DEF_TYPEID(precomp_jacobian_s);

static void precomp_jacobian_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(precomp_jacobian_s, _data);

	nlop_free(data->nlop);

	xfree(data);
}


static void zprecomp_jacobian_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], _Complex float* dst[OO], int II, const long idims[II][N], const _Complex float* src[II], const long ddims[OO][II][N], _Complex float* jac[OO][II])
{
	auto data = CAST_DOWN(precomp_jacobian_s, _data);

	UNUSED(odims);
	UNUSED(idims);

	auto op = data->nlop;

	assert(nlop_get_nr_out_args(op) == OO);
	assert(nlop_get_nr_in_args(op) == II);

	unsigned long out_der_flag = 0;
	unsigned long in_der_flag = 0;

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			if (NULL != jac[o][i]) {

				out_der_flag = MD_SET(out_der_flag, o);
				in_der_flag = MD_SET(in_der_flag, i);
			}

	void* args[OO + II];
	for (int i = 0; i < OO; i++)
		args[i] = dst[i];

	for (int i = 0; i < II; i++)
		args[i + OO] = (void*)src[i];

	nlop_generic_apply_select_derivative_unchecked(op, OO + II, args, out_der_flag, in_der_flag);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			if (NULL != jac[o][i])
				linop_compute_matrix_zblock_diag(nlop_get_derivative(op, o, i), N, ddims[o][i], jac[o][i]);

	nlop_clear_derivatives(op);
}

struct nlop_s* nlop_zprecomp_jacobian_F(const struct nlop_s* nlop)
{
	PTR_ALLOC(struct precomp_jacobian_s, _data);
	SET_TYPEID(precomp_jacobian_s, _data);

	int OO = nlop_get_nr_out_args(nlop);
	int II = nlop_get_nr_in_args(nlop);

	assert(0 < OO);
	assert(0 < II);

	_data->nlop = nlop;


	int N = nlop_generic_domain(nlop, 0)->N;

	long nl_odims[OO][N];
	long nl_idims[II][N];

	for (int i = 0; i < II; i++) {

		assert(N == (int)nlop_generic_domain(nlop, i)->N);
		md_copy_dims(N, nl_idims[i], nlop_generic_domain(nlop, i)->dims);
	}

	for (int o = 0; o < OO; o++) {

		assert(N == (int)nlop_generic_codomain(nlop, o)->N);
		md_copy_dims(N, nl_odims[o], nlop_generic_codomain(nlop, o)->dims);
	}

	unsigned long diag_flags[OO][II];
	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			diag_flags[o][i] = 0;

	return nlop_zblock_diag_generic_create(CAST_UP(PTR_PASS(_data)), N, OO, nl_odims, II, nl_idims, diag_flags, zprecomp_jacobian_fun, precomp_jacobian_del);
}


static void zrprecomp_jacobian_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], complex float* dst[OO], int II, const long idims[II][N], const complex float* src[II], const long ddims[OO][II][N], complex float* jac[OO][II], complex float* jacc[OO][II])
{
	auto data = CAST_DOWN(precomp_jacobian_s, _data);

	UNUSED(odims);
	UNUSED(idims);

	auto op = data->nlop;

	assert(nlop_get_nr_out_args(op) == OO);
	assert(nlop_get_nr_in_args(op) == II);

	unsigned long out_der_flag = 0;
	unsigned long in_der_flag = 0;

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			if (NULL != jac[o][i]) {

				out_der_flag = MD_SET(out_der_flag, o);
				in_der_flag = MD_SET(in_der_flag, i);
			}

	void* args[OO + II];
	for (int i = 0; i < OO; i++)
		args[i] = dst[i];

	for (int i = 0; i < II; i++)
		args[i + OO] = (void*)src[i];

	nlop_generic_apply_select_derivative_unchecked(op, OO + II, args, out_der_flag, in_der_flag);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			if (NULL != jac[o][i])
				linop_compute_matrix_zrblock_diag(nlop_get_derivative(op, o, i), N, ddims[o][i], jac[o][i], jacc[o][i]);

	nlop_clear_derivatives(op);
}

struct nlop_s* nlop_zrprecomp_jacobian_F(const struct nlop_s* nlop)
{
	PTR_ALLOC(struct precomp_jacobian_s, _data);
	SET_TYPEID(precomp_jacobian_s, _data);

	int OO = nlop_get_nr_out_args(nlop);
	int II = nlop_get_nr_in_args(nlop);

	assert(0 < OO);
	assert(0 < II);

	_data->nlop = nlop;

	int N = nlop_generic_domain(nlop, 0)->N;

	long nl_odims[OO][N];
	long nl_idims[II][N];

	for (int i = 0; i < II; i++) {

		assert(N == (int)nlop_generic_domain(nlop, i)->N);
		md_copy_dims(N, nl_idims[i], nlop_generic_domain(nlop, i)->dims);
	}

	for (int o = 0; o < OO; o++) {

		assert(N == (int)nlop_generic_codomain(nlop, o)->N);
		md_copy_dims(N, nl_odims[o], nlop_generic_codomain(nlop, o)->dims);
	}

	unsigned long diag_flags[OO][II];
	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			diag_flags[o][i] = 0;
	
	bool holomorphic[OO][II];
	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			holomorphic[o][i] = false;

	return nlop_zrblock_diag_generic_create(CAST_UP(PTR_PASS(_data)), N, OO, nl_odims, II, nl_idims, diag_flags, holomorphic, zrprecomp_jacobian_fun, precomp_jacobian_del);
}
