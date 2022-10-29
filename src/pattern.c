/* Copyright 2013, 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */


#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/opts.h"



static const char help_str[] = "Compute sampling pattern from kspace";


int main_pattern(int argc, char* argv[argc])
{
	const char* ksp_file = NULL;
	const char* pat_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_OUTFILE(true, &pat_file, "pattern"),
	};

	unsigned long flags = COIL_FLAG;

	const struct opt_s opts[] = {

		OPT_ULONG('s', &flags, "bitmask", "Squash dimensions selected by bitmask"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long in_dims[N];
	long out_dims[N];

	complex float* kspace = load_cfl(ksp_file, N, in_dims);

	md_select_dims(N, ~flags, out_dims, in_dims);
	
	complex float* pattern = create_cfl(pat_file, N, out_dims);

	estimate_pattern(N, in_dims, flags, pattern, kspace);

	unmap_cfl(N, in_dims, kspace);
	unmap_cfl(N, out_dims, pattern);

	return 0;
}




