/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 *
 * various functions built around md_loop
 * No GPU support at the moment!
 */

#include <complex.h>

#include "num/multind.h"

#include "loop.h"


// typedef complex float (*sample_fun_t)(void* _data, const long pos[]);

struct sample_data {

	unsigned int N;
	const long* strs;
	complex float* out;
	void* data;
	sample_fun_t fun;
};

static void sample_kernel(void* _data, const long pos[])
{
	struct sample_data* data = _data;
	data->out[md_calc_offset(data->N, data->strs, pos)] = data->fun(data->data, pos);
}

void md_zsample(unsigned int N, const long dims[N], complex float* out, void* data, sample_fun_t fun)
{
	struct sample_data sdata;

	sdata.N = N;

	long strs[N];
	md_calc_strides(N, strs, dims, 1);	// we use size = 1 here
	sdata.strs = strs;

	sdata.out = out;
	sdata.data = data;
	sdata.fun = fun;

	md_loop(N, dims, &sdata, sample_kernel);
}

void md_parallel_zsample(unsigned int N, const long dims[N], complex float* out, void* data, sample_fun_t fun)
{
	struct sample_data sdata;

	sdata.N = N;

	long strs[N];
	md_calc_strides(N, strs, dims, 1);	// we use size = 1 here
	sdata.strs = strs;

	sdata.out = out;
	sdata.data = data;
	sdata.fun = fun;

	md_parallel_loop(N, dims, ~0u, &sdata, sample_kernel);
}


struct map_data {

	unsigned int N;
	const long* strs;
	const complex float* in;
	void* data;
	map_fun_data_t fun;
};


static complex float map_kernel(void* _data, const long pos[])
{
	struct map_data* data = _data;

	return data->fun(data->data, data->in[md_calc_offset(data->N, data->strs, pos)]);
}


static void md_zmap_const(unsigned int N, const long dims[N], complex float* out, const complex float* in, void* data, map_fun_data_t fun)
{
	struct map_data sdata;

	sdata.N = N;

	long strs[N];
	md_calc_strides(N, strs, dims, 1); // we use size = 1 here 
	sdata.strs = strs;

	sdata.in = in;
	sdata.data = data;
	sdata.fun = fun;

	md_zsample(N, dims, out, &sdata, map_kernel);
}



static complex float map_data_kernel(void* _data, complex float arg)
{
	map_fun_t fun = _data;

	return fun(arg);
}

void md_zmap(unsigned int N, const long dims[N], complex float* out, const complex float* in, map_fun_t fun)
{
	md_zmap_const(N, dims, out, in, (void*)fun, map_data_kernel);
}	



struct gradient_data {

	unsigned int N;
	const complex float* grad;
};

static complex float gradient_kernel(void* _data, const long pos[])
{
	struct gradient_data* data = _data;

	complex float val = 0.;

	for (unsigned int i = 0; i < data->N; i++)
		val += pos[i] * data->grad[i];

	return val;
}


void md_zgradient(unsigned int N, const long dims[N], complex float* out, const complex float grad[N])
{
	struct gradient_data data = { N, grad };
	md_zsample(N, dims, out, &data, gradient_kernel);
}




