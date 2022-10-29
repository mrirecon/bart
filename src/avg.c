/* Copyright 2014-2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2014 Frank Ong <frankong@berkeley.edu.
 * 2014, 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif

static const char help_str[] = "Calculates (weighted) average along dimensions specified by bitmask.";


int main_avg(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "bitmask"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool wavg = false;

	const struct opt_s opts[] = {

		OPT_SET('w', &wavg, "weighted average"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;

	long idims[N];
	complex float* data = load_cfl(in_file, N, idims);

	long odims[N];
	md_select_dims(N, ~flags, odims, idims);

	complex float* out = create_cfl(out_file, N, odims);

	(wavg ? md_zwavg : md_zavg)(N, idims, flags, out, data);

	unmap_cfl(N, idims, data);
	unmap_cfl(N, odims, out);

	return 0;
}


