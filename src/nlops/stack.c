/* Copyright 2021. Uecker Lab. University Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

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

