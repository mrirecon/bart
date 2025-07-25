/* Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016,2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */
 
#include <stdlib.h>

#include "misc/debug.h"
#include "misc/types.h"
#include "misc/misc.h"

#include "iter/vec.h"

#include "monitor.h"


void iter_monitor(struct iter_monitor_s* monitor, const struct vec_iter_s* ops, const float* x)
{
	if ((NULL != monitor) && (NULL != monitor->fun))
		monitor->fun(monitor, ops, x);
}

void iter_history(struct iter_monitor_s* monitor, const struct iter_history_s* hist)
{
	if ((NULL != monitor) && (NULL != monitor->record))
		monitor->record(monitor, hist);
}



struct monitor_default_s {

	iter_monitor_t super;

	long N;
	const float* image_truth;
	double it_norm;

	void* data;
	float (*objective)(const void* data, const float* x);
};

static DEF_TYPEID(monitor_default_s);


static void monitor_default_fun(struct iter_monitor_s* _data, const struct vec_iter_s* vops, const float* x)
{
	static int iter = 0;
	auto data = CAST_DOWN(monitor_default_s, _data);

	double err = -1.;
	double obj = -1.;

	long N = data->N;

	if (NULL != data->image_truth) {

		if (-1. == data->it_norm)
			data->it_norm = vops->norm(N, data->image_truth);

		float* x_err = vops->allocate(N);

		vops->sub(N, x_err, data->image_truth, x);
		err = vops->norm(N, x_err) / data->it_norm;

		vops->del(x_err);
	}

	if (NULL != data->objective)
		obj = data->objective(data->data, x);

	debug_printf(DP_INFO, "[Iter %04d] Objective: %f, Error: %f\n", ++iter, obj, err);

	data->super.obj = obj;
	data->super.err = err;
}

struct iter_monitor_s* iter_monitor_create(long N, const float* image_truth, void* data, float (*objective)(const void* data, const float* x))
{
	PTR_ALLOC(struct monitor_default_s, monitor);
	SET_TYPEID(monitor_default_s, monitor);

	monitor->N = N;
	monitor->image_truth = image_truth;
	monitor->it_norm = -1.;
	monitor->data = data;
	monitor->objective = objective;

	monitor->super.fun = monitor_default_fun;
	monitor->super.record = NULL;
	monitor->super.obj = -1.;
	monitor->super.err = -1.;

	return CAST_UP(PTR_PASS(monitor));
}


void monitor_iter6(struct monitor_iter6_s* monitor, int epoch, int batch, int num_batches, float objective, long NI, const float* x[NI], char* post_string)
{
	if ((NULL != monitor) && (NULL != monitor->fun))
		monitor->fun(monitor, epoch, batch, num_batches, objective, NI, x, post_string);
}
