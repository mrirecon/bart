/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2013, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2013	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/rand.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char* usage_str = "<input> <output>";
static const char* help_str = "Add noise with selected variance to input.";



int main_noise(int argc, char* argv[])
{
	float var = 1.;
	float spike = 1.;
	bool rvc = false;
	int rinit = -1;

	const struct opt_s opts[] = {

		{ 's', true, opt_int, &rinit, NULL },
		{ 'S', true, opt_float, &spike, NULL },
		{ 'r', false, opt_set, &rvc, "\treal-valued input" },
		{ 'n', true, opt_float, &var, "\tvariance\tDEFAULT: 1.0" },
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (-1 != rinit)
		num_rand_init(rinit);


	unsigned int N = DIMS;
	long dims[N];

	complex float* y = load_cfl(argv[1], N, dims);

	complex float* x = create_cfl(argv[2], N, dims);

	long T = md_calc_size(N, dims);

	// scale var for complex data
	if (!rvc)
		var = var / 2.f;

	float stdev = sqrtf(var);

	for (long i = 0; i < T; i++) {

		x[i] = y[i];

		if (spike >= uniform_rand())
			x[i] += stdev * gaussian_rand();

		if (rvc)
			x[i] = crealf(x[i]);
	}

	unmap_cfl(N, dims, y);
	unmap_cfl(N, dims, x);
	exit(0);
}




