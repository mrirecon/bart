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

static const char usage_str[] = "<bitmask> <input> <output>";
static const char help_str[] = "Calculates (weighted) average along dimensions specified by bitmask.";


int main_avg(int argc, char* argv[argc])
{
	bool wavg = false;

	const struct opt_s opts[] = {

		OPT_SET('w', &wavg, "weighted average"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;

	unsigned int flags = atoi(argv[1]);

	long idims[N];
	complex float* data = load_cfl(argv[2], N, idims);

	long odims[N];
	md_select_dims(N, ~flags, odims, idims);

	complex float* out = create_cfl(argv[3], N, odims);

	(wavg ? md_zwavg : md_zavg)(N, idims, flags, out, data);

	unmap_cfl(N, idims, data);
	unmap_cfl(N, odims, out);

	exit(0);
}


