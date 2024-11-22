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
#include "num/mpi_ops.h"
#include "num/ops_graph.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif


#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/checkpointing.h"

#include "stack.h"


struct stack_s {

	nlop_data_t super;

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

static void stack_der(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stack_s, _data);
	int II = data->II;

	assert(0 == o);
	assert(i < II);

	long (*idims)[II][data->N] = (void*)data->idims;
	long (*istrs)[II][data->N] = (void*)data->istrs;
	long (*pos)[II][data->N] = (void*)data->pos;

	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_copy2(data->N, (*idims)[i], data->ostrs, &(MD_ACCESS(data->N, data->ostrs, (*pos)[i], dst)), (*istrs)[i], src, CFL_SIZE);
}

static void stack_adj(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(stack_s, _data);
	int II = data->II;

	assert(0 == o);
	assert(i < II);

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

	// memory efficient in graph execution
	auto tmp = result->op;
	result->op = operator_nograph_wrapper(tmp);
	operator_free(tmp);

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

	nlop_data_t super;

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

	bool split_mpi;
};

DEF_TYPEID(stack_container_s);

static void stack_clear_der(const nlop_data_t* _data)
{
	auto d = CAST_DOWN(stack_container_s, _data);

	for (int i = 0; i < d->Nnlops; i++)
		nlop_clear_derivatives(d->nlops[i]);
}

#ifndef USE_CUDA

static int set_streams(int /*streams*/)
{
	return 1;
}

static int thread_id(void)
{
	return 0;
}

#else

static int set_streams(int streams)
{
	if (-1 < cuda_get_device_id())
		return MIN(streams, cuda_set_stream_level());

	return 1;
}

static int thread_id(void)
{
	if (cuda_is_stream_default())
		return 0;

	if (-1 < cuda_get_device_id())
		return cuda_get_stream_id();
	else
		return 0;
}
#endif



static void stack_container_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto d = CAST_DOWN(stack_container_s, _data);
	int Nnlops = d->Nnlops;

	int OO = nlop_get_nr_out_args(d->nlops[0]);
	int II = nlop_get_nr_in_args(d->nlops[0]);

	bool der_requested[II][OO];

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			der_requested[i][o]= nlop_der_requested(_data, i, o);

	int streams = (d->split_mpi) ? set_streams(Nnlops) : 1;

#pragma omp parallel num_threads(streams)
	for (int j = 0; j < Nnlops; j++) {

		if (d->split_mpi) {

			if (mpi_get_rank() != j % mpi_get_num_procs())
				continue;

			if ((1 < streams) && (thread_id() != (j / mpi_get_num_procs()) % streams))
				continue;
		}

		nlop_unset_derivatives(d->nlops[j]);
		nlop_set_derivatives(d->nlops[j], II, OO, der_requested);

		void* targs[N];

		for (int k = 0; k < d->OO + d->II; k++)
			targs[k] = (void*)(args[k]) + (d->offsets)[j][k];

		nlop_generic_apply_unchecked(d->nlops[j], N, targs);
	}

	if (d->split_mpi && (1 < mpi_get_num_procs()))  {

		for (int i = 0; i < Nnlops; i++) {

			for (int o = 0; o < d->OO; o++) {

				auto iov = nlop_generic_codomain(d->nlops[i], o);
				mpi_bcast2(iov->N, iov->dims, iov->strs, (void*)(args[o]) + (d->offsets)[i][o], (long)iov->size, i % mpi_get_num_procs());
			}
		}
	}
}

static void stack_container_der(const nlop_data_t* _data, int o, int i, complex float* _dst, const complex float* _src)
{
	const auto d = CAST_DOWN(stack_container_s, _data);

	int Nnlops = d->Nnlops;

	int streams = (d->split_mpi) ? set_streams(Nnlops) : 1;

#pragma omp parallel num_threads(streams)
	for (int j = 0; j < Nnlops; j++) {

		if (d->split_mpi) {

			if (mpi_get_rank() != j % mpi_get_num_procs())
				continue;

			if ((1 < streams) && (thread_id() != (j / mpi_get_num_procs()) % streams))
				continue;
		}

		complex float* dst = (void*)_dst + d->offsets[j][o];
		const complex float* src = (const void*)_src + d->offsets[j][d->OO + i];

		linop_forward_unchecked(nlop_get_derivative(d->nlops[j], o, i), dst, src);
	}

	if (d->split_mpi && (1 < mpi_get_num_procs()))  {

		for (int j = 0; j < Nnlops; j++) {

			complex float* dst = (void*)_dst + d->offsets[j][o];

			auto iov = nlop_generic_codomain(d->nlops[j], o);
			mpi_bcast2(iov->N, iov->dims, iov->strs, dst, (long)iov->size, j % mpi_get_num_procs());
		}
	}
}

static void stack_container_adj(const nlop_data_t* _data, int o, int i, complex float* _dst, const complex float* _src)
{
	const auto d = CAST_DOWN(stack_container_s, _data);

	int Nnlops = d->Nnlops;

	auto dom = nlop_generic_domain(d->nlops[0], i);

	int streams = (d->split_mpi) ? set_streams(Nnlops) : 1;

	complex float* tmp[streams];

	if (d->dup[i]) {

		md_clear(dom->N, dom->dims, _dst, dom->size);

		for(int j = 0; j < streams; j++) {

			tmp[j] = md_alloc_sameplace(dom->N, dom->dims, dom->size, _dst);
			md_clear(dom->N, dom->dims, tmp[j], dom->size);
		}
	}

	//required as gpu calls outside omp parallel region reset streams
	streams = (d->split_mpi) ? set_streams(Nnlops) : 1;

#pragma omp parallel num_threads(streams)
	for (int j = 0; j < Nnlops; j++) {

		if (d->split_mpi) {

			if (mpi_get_rank() != j % mpi_get_num_procs())
				continue;

			if ((1 < streams) && (thread_id() != (j / mpi_get_num_procs()) % streams))
				continue;
		}

		complex float* dst = (void*)_dst + d->offsets[j][i + d->OO];
		const complex float* src = (const void*)_src + d->offsets[j][o];

		if (d->dup[i]) {

			complex float* tmp2 = md_alloc_sameplace(dom->N, dom->dims, dom->size, tmp[thread_id() % streams]);

			linop_adjoint_unchecked(nlop_get_derivative(d->nlops[j], o, i), tmp2, src);
			md_zadd(dom->N, dom->dims, tmp[thread_id() % streams], tmp[thread_id() % streams], tmp2);

			md_free(tmp2);
		} else {

			linop_adjoint_unchecked(nlop_get_derivative(d->nlops[j], o, i), dst, src);
		}
	}

	for (int j = 0; d->dup[i] && j < streams; j++) {

		md_zadd(dom->N, dom->dims, _dst, _dst, tmp[j]);
		md_free(tmp[j]);
	}

	if (d->split_mpi && (1 < mpi_get_num_procs()))  {

		for (int j = 0; j < Nnlops && !d->dup[i]; j++) {

			auto iov = nlop_generic_domain(d->nlops[j], i);

			complex float* dst = (void*)_dst + d->offsets[j][i + d->OO];
			mpi_bcast2(iov->N, iov->dims, iov->strs, dst, (long)iov->size, j % mpi_get_num_procs());
		}

		auto iov = nlop_generic_domain(d->nlops[0], i);

		if (d->dup[i])
			mpi_reduce_zsum(iov->N, iov->dims, _dst, _dst);

	}
}

static void stack_container_nrm(const nlop_data_t* _data, int o, int i, complex float* _dst, const complex float* _src)
{
	const auto d = CAST_DOWN(stack_container_s, _data);

	int Nnlops = d->Nnlops;

	auto dom = nlop_generic_domain(d->nlops[0], i);

	int streams = (d->split_mpi) ? set_streams(Nnlops) : 1;

	complex float* tmp[streams];

	if (d->dup[i]) {

		md_clear(dom->N, dom->dims, _dst, dom->size);

		for(int j = 0; j < streams; j++) {

			tmp[j] = md_alloc_sameplace(dom->N, dom->dims, dom->size, _dst);
			md_clear(dom->N, dom->dims, tmp[j], dom->size);
		}
	}

	//required as gpu calls outside omp parallel region reset streams
	streams = (d->split_mpi) ? set_streams(Nnlops) : 1;

#pragma omp parallel num_threads(streams)
	for (int j = 0; j < Nnlops; j++) {

		if (d->split_mpi) {

			if (mpi_get_rank() != j % mpi_get_num_procs())
				continue;

			if ((1 < streams) && (thread_id() != (j / mpi_get_num_procs()) % streams))
				continue;
		}

		complex float* dst = (void*)_dst + d->offsets[j][i + d->OO];
		const complex float* src = (const void*)_src + d->offsets[j][i + d->OO];

		if (d->dup[i]) {

			complex float* tmp2 = md_alloc_sameplace(dom->N, dom->dims, dom->size, tmp[thread_id() % streams]);

			linop_adjoint_unchecked(nlop_get_derivative(d->nlops[j], o, i), tmp2, src);
			md_zadd(dom->N, dom->dims, tmp[thread_id() % streams], tmp[thread_id() % streams], tmp2);

			md_free(tmp2);
		} else {

			linop_normal_unchecked(nlop_get_derivative(d->nlops[j], o, i), dst, src);
		}
	}

	for (int j = 0; d->dup[i] && j < streams; j++) {

		md_zadd(dom->N, dom->dims, _dst, _dst, tmp[j]);
		md_free(tmp[j]);
	}

	if (d->split_mpi && (1 < mpi_get_num_procs())) {

		for (int j = 0; j < Nnlops && !d->dup[i]; j++) {

			auto iov = nlop_generic_domain(d->nlops[j], i);

			complex float* dst = (void*)_dst + d->offsets[j][i + d->OO];
			mpi_bcast2(iov->N, iov->dims, iov->strs, dst, (long)iov->size, j % mpi_get_num_procs());
		}

		auto iov = nlop_generic_domain(d->nlops[0], i);

		if (d->dup[i])
			mpi_reduce_zsum(iov->N, iov->dims, _dst, _dst);

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


static const struct nlop_s* nlop_stack_container_internal_create(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO], bool split_mpi)
{
	PTR_ALLOC(struct stack_container_s, d);
	SET_TYPEID(stack_container_s, d);


	int max_DI = 0;
	int max_DO = 0;

	int DI[II];
	int DO[OO];

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


	d->dup = *TYPE_ALLOC(bool[II]);

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

		const struct nlop_s* tmp;

		// checkpointing container applies derivatives jointly
		// makes use of shared ops

		if (!nlop_is_checkpoint(nlops[i]) && ((II > 1) || (OO > 1)))
			tmp = nlop_checkpoint_create(nlops[i], true, false);
		else
			tmp = nlop_clone(nlops[i]);

		d->nlops[i] = nlop_copy_wrapper(OO, (const long**)d->strs, II, (const long**)d->strs + OO, tmp);

		nlop_free(tmp);
	}

	d->simple_flatten_in =     (1 == II)
				&& (in_stack_dim[0] >= 0)
				&& (   (in_stack_dim[0] == max_DI - 1)
				    || (1 == md_calc_size(max_DI - in_stack_dim[0] - 1, nl_idims[0] + in_stack_dim[0] + 1)));

	d->simple_flatten_out =    (1 == OO)
				&& (out_stack_dim[0] >= 0)
				&& (   (out_stack_dim[0] == max_DO - 1)
				    || (1 == md_calc_size(max_DO - out_stack_dim[0] - 1, nl_odims[0] + out_stack_dim[0] + 1)));

	d->split_mpi = split_mpi;// && 1 < mpi_get_num_procs();

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

static const struct nlop_s* nlop_stack_container_internal_create_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO], bool split_mpi)
{
	auto result = nlop_stack_container_internal_create(N, nlops, II, in_stack_dim, OO, out_stack_dim, split_mpi);

	for (int i = 0; i < N; i++)
		nlop_free(nlops[i]);

	return result;
}

const struct nlop_s* nlop_stack_container_create_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO])
{
	return nlop_stack_container_internal_create_F(N, nlops, II, in_stack_dim, OO, out_stack_dim, false);
}

const struct nlop_s* nlop_stack_multigpu_create_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO])
{
	return nlop_stack_container_internal_create_F(N, nlops, II, in_stack_dim, OO, out_stack_dim, true);
}

struct stack_flatten_trafo_s {

	linop_data_t super;

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
					assert(md_check_equal_dims(iov->N, st_dims[i], iov->dims, ~0UL));
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
			(*ioff)[i][j] = (-1 == in_stack_dim[i]) ? 0 : pos * strs[in_stack_dim[i]] / (long)CFL_SIZE;
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
	memset(st_dims, 0, sizeof st_dims);	// -fanalyzer uninitialized
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
			(*ooff)[i][j] = pos * strs[out_stack_dim[i]] / (long)CFL_SIZE;
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

	auto result = nlop_stack_container_internal_create_F(data->Nnlops, nlops, 1, istack_dims, 1, ostack_dims, data->split_mpi);

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
