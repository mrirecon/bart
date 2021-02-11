/* Copyright 2014. The Regents of the University of California.
 * Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 *
 * various functions built around md_loop
 * No GPU support at the moment!
 */

#include <complex.h>

#include "num/multind.h"

#include "misc/nested.h"

#include "loop.h"


// typedef complex float (*sample_fun_t)(const long pos[]);

void md_zsample(unsigned int N, const long dims[N], complex float* out, sample_fun_t fun)
{
	long strs[N];
	md_calc_strides(N, strs, dims, 1);	// we use size = 1 here

	long* strsp = strs;	// because of clang

	NESTED(void, sample_kernel, (const long pos[]))
	{
		out[md_calc_offset(N, strsp, pos)] = fun(pos);
	};

	md_loop(N, dims, sample_kernel);
}


void md_parallel_zsample(unsigned int N, const long dims[N], complex float* out, sample_fun_t fun)
{
	long strs[N];
	md_calc_strides(N, strs, dims, 1);	// we use size = 1 here

	long* strsp = strs;	// because of clang

	NESTED(void, sample_kernel, (const long pos[]))
	{
		out[md_calc_offset(N, strsp, pos)] = fun(pos);
	};

	md_parallel_loop(N, dims, ~0u, sample_kernel);
}




void md_zmap(unsigned int N, const long dims[N], complex float* out, const complex float* in, map_fun_t fun)
{
	long strs[N];
	md_calc_strides(N, strs, dims, 1); // we use size = 1 here 

	long* strsp = strs; // because of clang

	NESTED(complex float, map_kernel, (const long pos[]))
	{
		return fun(in[md_calc_offset(N, strsp, pos)]);
	};

	md_zsample(N, dims, out, map_kernel);
}







void md_zgradient(unsigned int N, const long dims[N], complex float* out, const complex float grad[N])
{
#if 1
	// clang
	const complex float* grad2 = grad;
#endif
	NESTED(complex float, gradient_kernel, (const long pos[]))
	{
		complex float val = 0.;

		for (int i = 0; i < (int)N; i++)
			val += pos[i] * grad2[i];

		return val;
	};

	md_zsample(N, dims, out, gradient_kernel);
}




