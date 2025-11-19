/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"


static const char help_str[] = "Create an array counting from 0 to {size-1} in dimensions {dim}.";



int main_index(int argc, char* argv[argc])
{
	int N = -1;
	int s = -1;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &N, "dim"),
		ARG_INT(true, &s, "size"),
		ARG_OUTFILE(true, &out_file, "name"),
	};

	bool inclusive = false;
	bool logspacing = false;
	float min = NAN;
	float max = NAN;

	const struct opt_s opts[] = {

		OPTL_SET(0, "end", &inclusive, "include the endpoint (last value is size)"),
		OPTL_SET(0, "log", &logspacing, "use logarithmic spacing (but specify min and max)"),
		OPTL_FLOAT(0, "min", &min, "min", "minimum value (default 0)"),
		OPTL_FLOAT(0, "max", &max, "max", "maximum value (default size)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (safe_isnanf(min))
		min = 0.0f;

	if (safe_isnanf(max))
		max = (float)(s);

	if (logspacing) {

		if (0 >= min)
			error("logspacing requires min > 0\n");

		min = logf(min);
		max = logf(max);
	}

	float dt = (max - min) / (inclusive ? (float)(s - 1) : (float)(s));

	num_init();

	assert(N >= 0);
	assert(s >= 0);

	long dims[N + 1];

	for (int i = 0; i < N; i++)
		dims[i] = 1;

	dims[N] = s;

	complex float* x = create_cfl(out_file, N + 1, dims);

	for (int i = 0; i < s; i++)
		x[i] = logspacing ? expf(min + i * dt) : (min + i * dt);

	unmap_cfl(N + 1, dims, x);

	return 0;
}

