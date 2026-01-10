/* Copyright 2013-2016. The Regents of the University of California.
 * Copyright 2015-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker
 * 2013	Jonathan Tamir
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/version.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Add noise with selected variance to input.";



int main_noise(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;


	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	float var = 1.f;
	float spike = 1.f;
	bool rvc = false;
	bool uniform = false;
	unsigned long long randseed = 0;

	const struct opt_s opts[] = {

		OPT_ULLONG('s', &randseed, "", "random seed initialization. '0' uses the default seed."),
		OPT_FLOAT('S', &spike, "", "()"),
		OPT_SET('r', &rvc, "real-valued input"),
		OPT_FLOAT('n', &var, "variance", "DEFAULT: 1.0"),
		OPTL_SET(0, "uniform", &uniform, "select uniform noise distribution"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	num_rand_init(randseed);


	int N = DIMS;
	long dims[N];

	complex float* y = load_cfl(in_file, N, dims);

	complex float* x = create_cfl(out_file, N, dims);

	// scale var for complex data
	if (!rvc)
		var = var / 2.f;

	float stdev = (uniform) ? 1. : sqrtf(var);

	if (use_compat_to_version("v0.9.00")) {

		long T = md_calc_size(N, dims);

		for (long i = 0; i < T; i++) {

			x[i] = y[i];

			if (spike >= uniform_rand())
				x[i] += stdev * gaussian_rand();

			if (rvc)
				x[i] = crealf(x[i]);
		}

	} else {

		md_copy(N, dims, x, y, CFL_SIZE);

		complex float* noise = md_alloc(N, dims, CFL_SIZE);

		if (uniform) // FIXME: uniform noise \in [0,0.7]?
			md_uniform_rand(N, dims, noise);
		else
			md_gaussian_rand(N, dims, noise);


		if (1.f != spike) {

			complex float* mask = md_alloc(N, dims, CFL_SIZE);
			md_rand_one(N, dims, mask, spike);

			md_zmul(N, dims, noise, noise, mask);

			md_free(mask);
		}

		md_zaxpy(N, dims, x, stdev, noise);

		md_free(noise);

		if (rvc)
			md_zreal(N, dims, x, x);
	}


	unmap_cfl(N, dims, y);
	unmap_cfl(N, dims, x);

	return 0;
}

