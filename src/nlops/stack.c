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
#include "num/ops.h"
#include "num/ops_graph.h"


#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/const.h"

#include "stack.h"


struct stack_s {

	INTERFACE(nlop_data_t);

	int N;
	int II;

	long* odims;
	long* ostrs;
	long* idims;
	long* istrs;
	long* pos;
};

DEF_TYPEID(stack_s);


static void stack_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(stack_s, _data);
	int II = data->II;

	assert(II + 1 == N);

	long (*idims)[II][data->N] = (void*)data->idims;
	long (*istrs)[II][data->N] = (void*)data->istrs;
	long (*pos)[II][data->N] = (void*)data->pos;

	for (int i = 0; i < II; i++)
		md_copy2(data->N, (*idims)[i], data->ostrs, &(MD_ACCESS(data->N, data->ostrs, (*pos)[i], args[0])), (*istrs)[i], args[i + 1], CFL_SIZE);

}

static void stack_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stack_s, _data);
	int II = data->II;

	assert(0 == o);
	assert((int)i < II);

	long (*idims)[II][data->N] = (void*)data->idims;
	long (*istrs)[II][data->N] = (void*)data->istrs;
	long (*pos)[II][data->N] = (void*)data->pos;

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_copy2(data->N, (*idims)[i], data->ostrs, &(MD_ACCESS(data->N, data->ostrs, (*pos)[i], dst)), (*istrs)[i], src, CFL_SIZE);
}

static void stack_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stack_s, _data);
	int II = data->II;

	assert(0 == o);
	assert((int)i < II);

	long (*idims)[II][data->N] = (void*)data->idims;
	long (*istrs)[II][data->N] = (void*)data->istrs;
	long (*pos)[II][data->N] = (void*)data->pos;

	md_copy2(data->N, (*idims)[i], (*istrs)[i], dst, data->ostrs, &(MD_ACCESS(data->N, data->ostrs, (*pos)[i], src)), CFL_SIZE);
}


static void stack_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(stack_s, _data);

	xfree(data->idims);
	xfree(data->odims);

	xfree(data->istrs);
	xfree(data->ostrs);

	xfree(data->pos);

	xfree(data);
}


struct nlop_s* nlop_stack_generic_create(int II, int N, const long odims[N], const long idims[II][N], int stack_dim)
{
	assert(stack_dim < N);
	assert(0 <=stack_dim);

	PTR_ALLOC(struct stack_s, data);
	SET_TYPEID(stack_s, data);

	data->N = N;
	data->II = II;

	data->odims = *TYPE_ALLOC(long[N]);
	data->ostrs = *TYPE_ALLOC(long[N]);
	data->idims = *TYPE_ALLOC(long[N * II]);
	data->istrs = *TYPE_ALLOC(long[N * II]);
	data->pos = *TYPE_ALLOC(long[N * II]);

	long (*tidims)[II][data->N] = (void*)data->idims;
	long (*tistrs)[II][data->N] = (void*)data->istrs;
	long (*tpos)[II][data->N] = (void*)data->pos;

	md_copy_dims(N, data->odims, odims);
	md_calc_strides(N, data->ostrs, odims, CFL_SIZE);

	long stack_size = 0;

	nlop_der_fun_t der [II][1];
	nlop_der_fun_t adj [II][1];

	for (int i = 0; i < II; i++) {

		md_copy_dims(N, (*tidims)[i], idims[i]);
		md_calc_strides(N, (*tistrs)[i], idims[i], CFL_SIZE);
		md_singleton_strides(N, (*tpos)[i]);

		(*tpos)[i][stack_dim] = stack_size;

		stack_size += (*tidims)[i][stack_dim];

		assert(md_check_equal_dims(N, odims, idims[i], ~MD_BIT(stack_dim)));

		der[i][0] = stack_der;
		adj[i][0] = stack_adj;
	}

	assert(stack_size == odims[stack_dim]);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->odims);

	return nlop_generic_create(1, N, nl_odims, II, N, idims, CAST_UP(PTR_PASS(data)), stack_fun, der, adj, NULL, NULL, stack_del);
}


struct nlop_s* nlop_stack_create(int N, const long odims[N], const long idims1[N], const long idims2[N], int stack_dim)
{
	long idims[2][N];
	md_copy_dims(N, idims[0], idims1);
	md_copy_dims(N, idims[1], idims2);

	return nlop_stack_generic_create(2, N, odims, idims, stack_dim);
}



struct nlop_s* nlop_destack_generic_create(int OO, int N, const long odims[OO][N], const long idims[N], int stack_dim)
{
	assert(stack_dim < N);
	assert(0 <=stack_dim);

	long pos[N];
	md_singleton_strides(N, pos);

	auto result = nlop_del_out_create(N, idims);

	for (int i = 0; i < OO; i++) {

		result = nlop_combine_FF(result, nlop_from_linop_F(linop_extract_create(N, pos, odims[i], idims)));
		result = nlop_dup_F(result, 0, 1);

		pos[stack_dim] += odims[i][stack_dim];

		assert(md_check_equal_dims(N, odims[i], idims, ~MD_BIT(stack_dim)));
	}

	assert(pos[stack_dim] == idims[stack_dim]);

	return result;
}


struct nlop_s* nlop_destack_create(int N, const long odims1[N], const long odims2[N], const long idims[N], int stack_dim)
{
	long odims[2][N];
	md_copy_dims(N, odims[0], odims1);
	md_copy_dims(N, odims[1], odims2);

	return nlop_destack_generic_create(2, N, odims, idims, stack_dim);
}


struct stack_container_s {

	INTERFACE(nlop_data_t);

	int Nnlops;

	int N;
	int OO;
	int* ostack_dims;
	int II;
	int* istack_dims;

	int* D;
	long** strs;

	bool* dup;
	long** offsets;

	const struct nlop_s** nlops;
	const struct nlop_s** nlops_original;

	bool simple_flatten_in;
	bool simple_flatten_out;
};

DEF_TYPEID(stack_container_s);

static void stack_clear_der(const nlop_data_t* _data)
{
	auto d = CAST_DOWN(stack_container_s, _data);

	for (int i = 0; i < d->Nnlops; i++)
		nlop_clear_derivatives(d->nlops[i]);
}

static int sc_threads(const struct stack_container_s* d)
{
	UNUSED(d);

	return -1;
}


static void stack_container_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto d = CAST_DOWN(stack_container_s, _data);
	int Nnlops = d->Nnlops;

	int OO = nlop_get_nr_out_args(d->nlops[0]);
	int II = nlop_get_nr_in_args(d->nlops[0]);

	const struct operator_s* ops[Nnlops];
	void* nargs[Nnlops][N];

	for (int i = 0; i < Nnlops; i++) {

		ops[i] = d->nlops[i]->op;

		for (int j = 0; j < N; j++)
			nargs[i][j] = (void*)args[j] + (d->offsets)[i][j];
	}

	bool der_requested[II][OO];
	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			der_requested[i][o]= nlop_der_requested(_data, i, o);

	for (int i = 0; i < Nnlops; i++) {
	
		nlop_unset_derivatives(d->nlops[i]);
		nlop_set_derivatives(d->nlops[i], II, OO, der_requested);
	}

	operator_generic_apply_parallel_unchecked(Nnlops, ops, N, nargs, sc_threads(d));
}

static void stack_container_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(stack_container_s, _data);

	int Nnlops = d->Nnlops;

	const struct operator_s* der[Nnlops];
	complex float* ndst[Nnlops];
	const complex float* nsrc[Nnlops];

	for (int j = 0; j < Nnlops; j++) {

		der[j] = nlop_get_derivative(d->nlops[j], o, i)->forward;

		ndst[j] = (void*)dst + d->offsets[j][o];
		nsrc[j] = (const void*)src + d->offsets[j][d->OO + i];
	}

	operator_apply_parallel_unchecked(Nnlops, der, ndst, nsrc, sc_threads(d));
}

static void stack_container_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(stack_container_s, _data);

	int Nnlops = d->Nnlops;

	const struct operator_s* adj[Nnlops];
	complex float* ndst[Nnlops];
	const complex float* nsrc[Nnlops];

	for (int j = 0; j < Nnlops; j++) {

		adj[j] = nlop_get_derivative(d->nlops[j], o, i)->adjoint;

		auto cod = operator_codomain(adj[j]);

		ndst[j] = ((d->dup[i]) && (0 < j)) ? md_alloc_sameplace(cod->N, cod->dims, cod->size, dst) : (void*)dst + d->offsets[j][d->OO + i];
		nsrc[j] = (const void*)src + d->offsets[j][o];
	}

	operator_apply_parallel_unchecked(Nnlops, adj, ndst, nsrc, sc_threads(d));

	if (d->dup[i]) {

		for (int j = 1; j < Nnlops; j++) {

			auto iov = operator_codomain(adj[j]);
			md_zadd(iov->N, iov->dims, dst, dst, ndst[j]);
			md_free(ndst[j]);
		}
	}
}

static void stack_container_nrm(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(stack_container_s, _data);

	int Nnlops = d->Nnlops;

	const struct operator_s* nrm[Nnlops];
	complex float* ndst[Nnlops];
	const complex float* nsrc[Nnlops];

	for (int j = 0; j < Nnlops; j++) {

		nrm[j] = nlop_get_derivative(d->nlops[j], o, i)->normal;

		auto cod = operator_codomain(nrm[j]);

		ndst[j] = (d->dup[i] && 0 < j) ? md_alloc_sameplace(cod->N, cod->dims, cod->size, dst) : (void*)dst + d->offsets[j][d->OO + i];
		nsrc[j] = (const void*)src + d->offsets[j][d->OO + i];
	}

	operator_apply_parallel_unchecked(Nnlops, nrm, ndst, nsrc, sc_threads(d));

	if (d->dup[i]) {

		for (int j = 1; j < Nnlops; j++) {

			auto iov = operator_codomain(nrm[j]);
			md_zadd(iov->N, iov->dims, dst, dst, ndst[j]);
			md_free(ndst[j]);
		}
	}
}



static void stack_container_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(stack_container_s, _data);

	for (int i = 0; i < d->Nnlops; i++) {

		xfree(d->offsets[i]);
		nlop_free(d->nlops[i]);
		nlop_free(d->nlops_original[i]);
	}

	for (int i = 0; i < d->N; i++)
		xfree(d->strs[i]);

	xfree(d->D);

	xfree(d->ostack_dims);
	xfree(d->istack_dims);

	xfree(d->dup);
	xfree(d->strs);
	xfree(d->offsets);
	xfree(d->nlops);
	xfree(d->nlops_original);

	xfree(d);
}

typedef const struct graph_s* (*nlop_graph_t)(const struct operator_s* op, const nlop_data_t* _data);

static const struct graph_s* nlop_graph_stack_container(const struct operator_s* op, const nlop_data_t* _data)
{
	auto d = CAST_DOWN(stack_container_s, _data);

	auto subgraph = operator_get_graph(d->nlops[0]->op);
	auto result = create_graph_container(op, "stack container", subgraph);

	return result;
}


const struct nlop_s* nlop_stack_container_create(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO])
{
	PTR_ALLOC(struct stack_container_s, d);
	SET_TYPEID(stack_container_s, d);

	int max_DI = 0;
	int max_DO = 0;

	unsigned int DI[II];
	unsigned int DO[OO];

	for (int i = 0; i < OO; i++) {

		auto iov = nlop_generic_codomain(nlops[0], i);
		DO[i] = iov->N;
		max_DO = MAX(max_DO, iov->N);
	}
	

	for (int i = 0; i < II; i++) {

		auto iov = nlop_generic_domain(nlops[0], i);
		DI[i] = iov->N;
		max_DI = MAX(max_DI, iov->N);
	}

	d->Nnlops = N;
	d->nlops = *TYPE_ALLOC(const struct nlop_s*[N]);
	d->nlops_original = *TYPE_ALLOC(const struct nlop_s*[N]);

	d->offsets = *TYPE_ALLOC(long*[N]);

	for (int i = 0; i < N; i++)
		d->offsets[i] = *TYPE_ALLOC(long[II + OO]);


	d->II = II;
	d->OO = OO;
	d->N = II + OO;


	d->dup = *TYPE_ALLOC(_Bool[II]);

	for (int i = 0; i < II; i++)
		d->dup[i] = (-1 == in_stack_dim[i]);


	d->D = *TYPE_ALLOC(int[II + OO]);
	d->strs = *TYPE_ALLOC(long*[II + OO]);

	d->istack_dims = ARR_CLONE(int[II], in_stack_dim);
	d->ostack_dims = ARR_CLONE(int[OO], out_stack_dim);

	long nl_odims[OO][max_DO];
	long nl_idims[II][max_DI];


	for (int i = 0; i < OO; i++) {

		md_singleton_dims(max_DO, nl_odims[i]);
		auto iov = nlop_generic_codomain(nlops[0], i);
		md_copy_dims(iov->N, nl_odims[i], iov->dims);
	}

	for (int i = 0; i < II; i++) {

		md_singleton_dims(max_DI, nl_idims[i]);
		auto iov = nlop_generic_domain(nlops[0], i);
		md_copy_dims(iov->N, nl_idims[i], iov->dims);
	}

	for (int j = 1; j < N; j++) {

		for (int i = 0; i < OO; i++) {

			auto iov = nlop_generic_codomain(nlops[j], i);
			assert(md_check_equal_dims(DO[i], nl_odims[i], iov->dims, ~MD_BIT(out_stack_dim[i])));
			nl_odims[i][out_stack_dim[i]] += iov->dims[out_stack_dim[i]];
		}

		for (int i = 0; i < II; i++) {

			auto iov = nlop_generic_domain(nlops[j], i);

			if (-1 == in_stack_dim[i]) {

				assert(iovec_check(iov, DI[i], nl_idims[i], MD_STRIDES(DI[i], nl_idims[i], CFL_SIZE)));
 			} else {

				assert(md_check_equal_dims(DI[i], nl_idims[i], iov->dims, ~MD_BIT(in_stack_dim[i])));
				nl_idims[i][in_stack_dim[i]] += iov->dims[in_stack_dim[i]];
			}
		}
	}


	for (int i = 0; i < OO; i++) {

		d->D[i] = DO[i];
		d->strs[i] = *TYPE_ALLOC(long[DO[i]]);
		md_calc_strides(DO[i], d->strs[i], nl_odims[i], CFL_SIZE);

		long pos[DO[i]];
		md_singleton_strides(DO[i], pos);

		int sd = out_stack_dim[i];

		for (int j = 0; j < N; j++) {

			d->offsets[j][i] = md_calc_offset(DO[i], d->strs[i], pos);
			if (0 <= sd) {

				auto iov = nlop_generic_codomain(nlops[j], i);
				pos[sd] += iov->dims[sd];
			}
				
		}
	}

	for (int i = 0; i < II; i++) {

		d->D[i + OO] = DI[i];
		d->strs[i + OO] = *TYPE_ALLOC(long[DI[i]]);
		md_calc_strides(DI[i], d->strs[i + OO], nl_idims[i], CFL_SIZE);

		long pos[DI[i]];
		md_singleton_strides(DI[i], pos);

		int sd = in_stack_dim[i];

		for (int j = 0; j < N; j++) {

			d->offsets[j][i + OO] = md_calc_offset(DI[i], d->strs[OO + i], pos);
			if (0 <= sd) {

				auto iov = nlop_generic_domain(nlops[j], i);
				pos[sd] += iov->dims[sd];
			}			
		}
	}

	for (int i = 0; i < N; i++) {

		d->nlops_original[i] = nlop_clone(nlops[i]);
		d->nlops[i] = nlop_copy_wrapper(OO, (const long**)d->strs, II, (const long**)d->strs + OO, nlops[i]);
	}

	d->simple_flatten_in =     (1 == II)
				&& (in_stack_dim[0] >= 0)
				&& (   (in_stack_dim[0] == (int)max_DI - 1)
				    || (1 == md_calc_size(max_DI - in_stack_dim[0] - 1, nl_idims[0] + in_stack_dim[0] + 1)));
				
	d->simple_flatten_out =    (1 == OO)
				&& (out_stack_dim[0] >= 0)
				&& (   (out_stack_dim[0] == (int)max_DO - 1)
				    || (1 == md_calc_size(max_DO - out_stack_dim[0] - 1, nl_odims[0] + out_stack_dim[0] + 1)));


	nlop_der_fun_t der_funs[II][OO];
	nlop_der_fun_t adj_funs[II][OO];
	nlop_der_fun_t nrm_funs[II][OO];

	for(int i = 0; i < II; i++)
		for(int o = 0; o < OO; o++) {

			der_funs[i][o] = stack_container_der;
			adj_funs[i][o] = stack_container_adj;
			nrm_funs[i][o] = stack_container_nrm;
		}

	const struct nlop_s* result = nlop_generic_managed_create(	OO, max_DO, nl_odims, II, max_DI, nl_idims, CAST_UP(PTR_PASS(d)),
									stack_container_fun, der_funs, adj_funs, nrm_funs, NULL, stack_container_del, stack_clear_der, nlop_graph_stack_container);


	for(int i = 0; i < II; i++)
		for(int o = 0; o < OO; o++) {

			bool null_op = true;
			for (int j = 0; j < N; j++)
				null_op = null_op && linop_is_null(nlop_get_derivative(nlops[j], o, i));

			if (null_op)
				result = nlop_no_der_F(result, o, i);
		}
			

	for (int i = 0; i < II; i++)
		result = nlop_reshape_in_F(result, i, DI[i], nl_idims[i]);
	
	for (int i = 0; i < OO; i++)
		result = nlop_reshape_out_F(result, i, DO[i], nl_odims[i]);
	
	return result;
}

const struct nlop_s* nlop_stack_container_create_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO])
{
	auto result = nlop_stack_container_create(N, nlops, II, in_stack_dim, OO, out_stack_dim);

	for (int i = 0; i < N; i++)
		nlop_free(nlops[i]);
	
	return result;
}

struct stack_flatten_trafo_s {

	INTERFACE(linop_data_t);

	long isize;
	long osize;

	int N;
	
	long* ioff;
	long* ooff;

	int* D;
	long** dims;
	long** ostrs;
	long** istrs;

	bool dup;
};

static DEF_TYPEID(stack_flatten_trafo_s);


static void stack_flatten_trafo_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(stack_flatten_trafo_s, _data);

	for (int i = 0; i < d->N; i++)
		md_copy2(d->D[i], d->dims[i], d->ostrs[i], dst + d->ooff[i], d->istrs[i], src + d->ioff[i], CFL_SIZE);
}

static void stack_flatten_trafo_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(stack_flatten_trafo_s, _data);

	if (d->dup) {

		md_clear(1, MD_DIMS(d->isize), dst, CFL_SIZE);
		for (int i = 0; i < d->N; i++)
			md_zadd2(d->D[i], d->dims[i], d->istrs[i], dst + d->ioff[i], d->istrs[i], dst + d->ioff[i], d->ostrs[i], src + d->ooff[i]);

	} else {

		for (int i = 0; i < d->N; i++)
			md_copy2(d->D[i], d->dims[i], d->istrs[i], dst + d->ioff[i], d->ostrs[i], src + d->ooff[i], CFL_SIZE);
	}
}

static void stack_flatten_trafo_free(const linop_data_t* _data)
{
	const auto d = CAST_DOWN(stack_flatten_trafo_s, _data);

	for (int i = 0; i < d->N; i++) {

		xfree(d->dims[i]);
		xfree(d->istrs[i]);
		xfree(d->ostrs[i]);
	}

	xfree(d->D);
	xfree(d->dims);
	xfree(d->ostrs);
	xfree(d->istrs);
	xfree(d->ooff);
	xfree(d->ioff);

	xfree(d);
}


static struct nlop_s* nlop_flatten_stacked_transform_input_create(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II])
{
	PTR_ALLOC(struct stack_flatten_trafo_s, d);
	SET_TYPEID(stack_flatten_trafo_s, d);

	d->N = N * II;

	d->D = *TYPE_ALLOC(int[d->N]);
	d->dims = *TYPE_ALLOC(long*[d->N]);
	d->ostrs = *TYPE_ALLOC(long*[d->N]);
	d->istrs = *TYPE_ALLOC(long*[d->N]);
	d->ooff = *TYPE_ALLOC(long[d->N]);
	d->ioff = *TYPE_ALLOC(long[d->N]);

	int (*D)[II][N] = (void*)d->D;
	long* (*dims)[II][N] = (void*)d->dims;
	long* (*ostrs)[II][N] = (void*)d->ostrs;
	long* (*istrs)[II][N] = (void*)d->istrs;
	long (*ooff)[II][N] = (void*)d->ooff;
	long (*ioff)[II][N] = (void*)d->ioff;

	long* st_dims[II];
	
	for (int i = 0; i < II; i++) {

		for (int j = 0; j < N; j++) {

			auto iov = nlop_generic_domain(nlops[j], i);

			(*D)[i][j] = iov->N;
			(*dims)[i][j] = ARR_CLONE(long[iov->N], iov->dims);
			(*ostrs)[i][j] = ARR_CLONE(long[iov->N], iov->strs);

			if (0 == j) {
				st_dims[i] = ARR_CLONE(long[iov->N], iov->dims);

			} else {

				if (-1 == in_stack_dim[i])
					assert(md_check_equal_dims(iov->N, st_dims[i], iov->dims, ~0));
				else {
					assert(md_check_equal_dims(iov->N, st_dims[i], iov->dims, ~MD_BIT(in_stack_dim[i])));
					st_dims[i][in_stack_dim[i]] += iov->dims[in_stack_dim[i]];
				}
			}
		}
	}

	for (int i = 0; i < II; i++) {

		long pos = 0;
		long strs[(*D)[i][0]];
		md_calc_strides((*D)[i][0], strs, st_dims[i], CFL_SIZE);

		for (int j = 0; j < N; j++) {

			(*istrs)[i][j] = ARR_CLONE(long[(*D)[i][j]], strs);
			(*ioff)[i][j] = (-1 == in_stack_dim[i]) ? 0 : pos * strs[in_stack_dim[i]] / CFL_SIZE;
			pos += (-1 == in_stack_dim[i]) ? 0 : (*dims)[i][j][in_stack_dim[i]];
		}
	}

	d->isize = 0;

	for (int i = 0; i < II; i++) {

		for (int j = 0; j < N; j++)
			(*ioff)[i][j] += d->isize;

		d->isize += md_calc_size((*D)[i][0], st_dims[i]);
		xfree(st_dims[i]);
	}

	d->osize = 0;

	for (int j = 0; j < N; j++)
		for (int i = 0; i < II; i++) {

			(*ooff)[i][j] = d->osize;
			d->osize += md_calc_size((*D)[i][j], (*dims)[i][j]);
		}
	
	d->dup = false;
	for (int i = 0; i < II; i++)
		d->dup = d->dup || (-1 == in_stack_dim[i]);

	long odims[1] = { d->osize };
	long idims[1] = { d->isize };

	return nlop_from_linop_F(linop_create(1, odims, 1, idims, CAST_UP(PTR_PASS(d)), stack_flatten_trafo_apply, stack_flatten_trafo_adjoint, NULL, NULL, stack_flatten_trafo_free));
}



static struct nlop_s* nlop_flatten_stacked_transform_output_create(int N, const struct nlop_s* nlops[N], int OO, int out_stack_dim[OO])
{
	PTR_ALLOC(struct stack_flatten_trafo_s, d);
	SET_TYPEID(stack_flatten_trafo_s, d);

	d->N = N * OO;

	d->D = *TYPE_ALLOC(int[d->N]);
	d->dims = *TYPE_ALLOC(long*[d->N]);
	d->ostrs = *TYPE_ALLOC(long*[d->N]);
	d->istrs = *TYPE_ALLOC(long*[d->N]);
	d->ooff = *TYPE_ALLOC(long[d->N]);
	d->ioff = *TYPE_ALLOC(long[d->N]);

	int (*D)[OO][N] = (void*)d->D;
	long* (*dims)[OO][N] = (void*)d->dims;
	long* (*ostrs)[OO][N] = (void*)d->ostrs;
	long* (*istrs)[OO][N] = (void*)d->istrs;
	long (*ooff)[OO][N] = (void*)d->ooff;
	long (*ioff)[OO][N] = (void*)d->ioff;

	long* st_dims[OO];
	
	for (int i = 0; i < OO; i++) {

		for (int j = 0; j < N; j++) {

			auto iov = nlop_generic_codomain(nlops[j], i);

			(*D)[i][j] = iov->N;
			(*dims)[i][j] = ARR_CLONE(long[iov->N], iov->dims);
			(*istrs)[i][j] = ARR_CLONE(long[iov->N], iov->strs);

			if (0 == j) {
				st_dims[i] = ARR_CLONE(long[iov->N], iov->dims);

			} else {

				assert(0 <= out_stack_dim[i]);
				assert(md_check_equal_dims(iov->N, st_dims[i], iov->dims, ~MD_BIT(out_stack_dim[i])));
				st_dims[i][out_stack_dim[i]] += iov->dims[out_stack_dim[i]];
			}
		}
	}

	for (int i = 0; i < OO; i++) {

		long pos = 0;
		long strs[(*D)[i][0]];
		md_calc_strides((*D)[i][0], strs, st_dims[i], CFL_SIZE);

		for (int j = 0; j < N; j++) {

			(*ostrs)[i][j] = ARR_CLONE(long[(*D)[i][j]], strs);
			(*ooff)[i][j] = pos * strs[out_stack_dim[i]] / CFL_SIZE;
			pos += (*dims)[i][j][out_stack_dim[i]];
		}
	}

	d->osize = 0;

	for (int i = 0; i < OO; i++) {

		for (int j = 0; j < N; j++)
			(*ooff)[i][j] += d->osize;

		d->osize += md_calc_size((*D)[i][0], st_dims[i]);
		xfree(st_dims[i]);
	}

	d->isize = 0;

	for (int j = 0; j < N; j++)
		for (int i = 0; i < OO; i++) {

			(*ioff)[i][j] = d->isize;
			d->isize += md_calc_size((*D)[i][j], (*dims)[i][j]);
		}
	
	d->dup = false;

	long odims[1] = { d->osize };
	long idims[1] = { d->isize };

	return nlop_from_linop_F(linop_create(1, odims, 1, idims, CAST_UP(PTR_PASS(d)), stack_flatten_trafo_apply, stack_flatten_trafo_adjoint, NULL, NULL, stack_flatten_trafo_free));
}

const struct nlop_s* nlop_flatten_stacked(const struct nlop_s* nlop)
{
	auto _data = nlop_get_data_nested(nlop);
	if (NULL == _data)
		return NULL;

	auto data = CAST_MAYBE(stack_container_s, _data);	
	if (NULL == data)
		return NULL;

	int N = data->Nnlops;
	int II = data->II;
	int OO = data->OO;

	auto itrafo = data->simple_flatten_in ? NULL : nlop_flatten_stacked_transform_input_create(N, data->nlops_original, II, data->istack_dims);
	auto otrafo = data->simple_flatten_out ? NULL : nlop_flatten_stacked_transform_output_create(N, data->nlops_original, OO, data->ostack_dims);

	const struct nlop_s* nlops[data->Nnlops];
	for (int i = 0; i < data->Nnlops; i++)
		nlops[i] = nlop_flatten(data->nlops_original[i]);

	int istack_dims[1] = { 0 };
	int ostack_dims[1] = { 0 };
	
	auto result = nlop_stack_container_create_F(data->Nnlops, nlops, 1, istack_dims, 1, ostack_dims);

	if (NULL != itrafo)
		result = nlop_chain_FF(itrafo, result);
	
	if (NULL == otrafo)
		return result;

	auto frw = operator_chain(nlop_get_derivative(result,0,0)->forward, nlop_get_derivative(otrafo,0,0)->forward);
	auto adj = operator_chain(nlop_get_derivative(otrafo,0,0)->adjoint, nlop_get_derivative(result,0,0)->adjoint);
	auto nrm = nlop_get_derivative(result,0,0)->normal;

	PTR_ALLOC(struct nlop_s, n);

	auto der = TYPE_ALLOC(const struct linop_s*[1][1]);
	(*der)[0][0] = linop_from_ops(frw, adj, nrm, NULL);
	n->derivative = &(*der)[0][0];

	n->op = operator_chain(result->op, otrafo->op);

	operator_free(frw);
	operator_free(adj);

	nlop_free(result);
	nlop_free(otrafo);

	return PTR_PASS(n);
}
