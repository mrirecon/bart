/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>

#include "misc/misc.h"
#include "misc/types.h"

#include "num/multind.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "iter/iter2.h"

#include "itop.h"


struct itop_s {

	INTERFACE(operator_data_t);

	italgo_fun2_t italgo;
	iter_conf* iconf;
	const struct operator_s* op;
	unsigned int num_funs;
	long size;

	const float* init;

	const struct operator_p_s** prox_funs;
	const struct linop_s** prox_linops;
};

DEF_TYPEID(itop_s);

static void itop_apply(const operator_data_t* _data, unsigned int N, void* args[static N])
{
	assert(2 == N);
	const struct itop_s* data = CAST_DOWN(itop_s, _data);

	if (NULL == data->init) {

		md_clear(1, MD_DIMS(data->size), args[0], sizeof(float));

	} else {

		const struct iovec_s* iov = operator_domain(data->op);

		md_copy(iov->N, iov->dims, args[0], data->init, iov->size);
	}

	data->italgo(data->iconf, data->op, data->num_funs, data->prox_funs, data->prox_linops, NULL, 
			NULL, data->size, args[0], args[1], NULL);
}

static void itop_del(const operator_data_t* _data)
{
	const struct itop_s* data = CAST_DOWN(itop_s, _data);

	operator_free(data->op);

	if (NULL != data->init)
		md_free(data->init);

	if (NULL != data->prox_funs) {

		for (unsigned int i = 0; i < data->num_funs; i++)
			operator_p_free(data->prox_funs[i]);

		xfree(data->prox_funs);
	}

	if (NULL != data->prox_linops) {

		for (unsigned int i = 0; i < data->num_funs; i++)
			linop_free(data->prox_linops[i]);

		xfree(data->prox_linops);
	}
	
	xfree(data);		
}


const struct operator_s* itop_create(	italgo_fun2_t italgo, iter_conf* iconf,
					const float* init,
					const struct operator_s* op,
					unsigned int num_funs,
					const struct operator_p_s* prox_funs[num_funs],
					const struct linop_s* prox_linops[num_funs])
{
	PTR_ALLOC(struct itop_s, data);
	SET_TYPEID(itop_s, data);

	const struct iovec_s* iov = operator_domain(op);

	data->iconf = iconf;
	data->italgo = italgo;
	data->op = operator_ref(op);
	data->num_funs = num_funs;
	data->size = 2 * md_calc_size(iov->N, iov->dims);	// FIXME: do not assume complex
	data->prox_funs = NULL;
	data->prox_linops = NULL;
	data->init = NULL;

	if (NULL != init) {

		float* init2 = md_alloc(iov->N, iov->dims, iov->size);
		md_copy(iov->N, iov->dims, init2, init, iov->size);
		data->init = init2;
	}

	if (NULL != prox_funs) {

		data->prox_funs = *TYPE_ALLOC(const struct operator_p_s*[num_funs]);

		for (unsigned int i = 0; i < num_funs; i++)
			data->prox_funs[i] = operator_p_ref(prox_funs[i]);
	}

	if (NULL != prox_linops) {

		data->prox_linops = *TYPE_ALLOC(const struct linop_s*[num_funs]);

		for (unsigned int i = 0; i < num_funs; i++)
			data->prox_linops[i] = linop_clone(prox_linops[i]);
	}

	return operator_create(iov->N, iov->dims, iov->N, iov->dims, CAST_UP(PTR_PASS(data)), itop_apply, itop_del);
}


