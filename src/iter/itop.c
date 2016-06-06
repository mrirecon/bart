/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "iter/iter2.h"

#include "itop.h"


struct itop_s {

	operator_data_t base;

	italgo_fun2_t italgo;
	iter_conf* iconf;
	const struct operator_s* op;
	unsigned int num_funs;
	long size;

	const struct operator_p_s** prox_funs;
	const struct linop_s** prox_linops;
};

static void itop_apply(const operator_data_t* _data, unsigned int N, void* args[static N])
{
	assert(2 == N);
	const struct itop_s* data = CONTAINER_OF(_data, const struct itop_s, base);

	md_clear(1, MD_DIMS(data->size), args[0], sizeof(float));
		
	data->italgo(data->iconf, data->op, data->num_funs, data->prox_funs, data->prox_linops, NULL, 
			data->size, args[0], args[1], NULL, NULL, NULL);
}

static void itop_del(const operator_data_t* _data)
{
	const struct itop_s* data = CONTAINER_OF(_data, const struct itop_s, base);

	operator_free(data->op);

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
					const struct operator_s* op,
					unsigned int num_funs,
					const struct operator_p_s* prox_funs[static num_funs],
					const struct linop_s* prox_linops[static num_funs])
{
	PTR_ALLOC(struct itop_s, data);

	const struct iovec_s* iov = operator_domain(op);

	data->iconf = iconf;
	data->italgo = italgo;
	data->op = operator_ref(op);
	data->num_funs = num_funs;
	data->size = 2 * md_calc_size(iov->N, iov->dims);	// FIXME: do not assume complex
	data->prox_funs = NULL;
	data->prox_linops = NULL;

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

	return operator_create(iov->N, iov->dims, iov->N, iov->dims, &PTR_PASS(data)->base, itop_apply, itop_del);
}


