/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Dara Bahri <dbahri123@gmail.com>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2015-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = 
	"Output normalized root mean square error (NRMSE),\n"
	"i.e. norm(input - ref) / norm(ref)";
			


int main_nrmse(int argc, char* argv[argc])
{
	const char* ref_file = NULL;
	const char* in_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ref_file, "reference"),
		ARG_INFILE(true, &in_file, "input"),
	};

	float test = -1.;
	bool auto_scale = false;
	bool scientific = false;

	const struct opt_s opts[] = {

		OPT_FLOAT('t', &test, "eps", "compare to eps"),
		OPT_SET('s', &auto_scale, "automatic (complex) scaling"),
		OPTL_SET('S', "scientific", &scientific, "use scientific notation in output"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long ref_dims[DIMS];
	long in_dims[DIMS];

	complex float* ref = load_cfl(ref_file, DIMS, ref_dims);
	complex float* in = load_cfl(in_file, DIMS, in_dims);

	assert(md_check_compat(DIMS, 0u, in_dims, ref_dims));

	if (auto_scale) {

		complex float sc = md_zscalar(DIMS, ref_dims, in, ref);

		float n = md_znorm(DIMS, ref_dims, ref);

		if (0. == n)
			error("Reference has zero norm");

		sc /= n * n;

		debug_printf(DP_INFO, "Scaled by: %f%+fi\n", crealf(sc), cimagf(sc));

		md_zsmul(DIMS, ref_dims, ref, ref, sc);
	}

	float err = md_znrmse(DIMS, ref_dims, ref, in);

	if (scientific)
		bart_printf("%e\n", err);
	else
		bart_printf("%f\n", err);

	unmap_cfl(DIMS, ref_dims, ref);
	unmap_cfl(DIMS, in_dims, in);

	if (test == -1.)
		test = err;

	if (err < test * 0.1)
		debug_printf(DP_DEBUG2, "Loose test: %f <= %f x %f\n", err, 0.1, test);

	return (err <= test) ? 0 : 1;
}



