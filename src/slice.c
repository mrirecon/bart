/* Copyright 2013-2016. The Regents of the University of California.
 * Copyright 2017-2021. Uecker Lab. Unversity Medical Center Göttingen.
 * Copyright 2021-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"



#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Extracts a slice from positions along dimensions.";


int main_slice(int argc, char* argv[argc])
{
	int count = 0;
	long* dims = NULL;
	long* poss = NULL;

	const char* in_file = NULL;
	const char* out_file = NULL;


	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 2, TUPLE_LONG(&dims, "dim"),
					   TUPLE_LONG(&poss, "pos")),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long in_dims[DIMS];
	long out_dims[DIMS];
	
	complex float* in_data = load_cfl(in_file, DIMS, in_dims);
	md_copy_dims(DIMS, out_dims, in_dims);

	long pos2[DIMS] = { [0 ... DIMS - 1] = 0 };
	unsigned long flags = 0L;

	for (int i = 0; i < count; i++) {

		if ((0 > dims[i]) || (dims[i] >= DIMS))
			error("Dimensions %d is out of range", i);

		if ((0 > poss[i]) || (poss[i] >= in_dims[dims[i]]))
			error("Position %d is out of range", i);

		out_dims[dims[i]] = 1;
		flags = MD_SET(flags, dims[i]);
		pos2[dims[i]] = poss[i];
	}


	complex float* out_data = create_cfl(out_file, DIMS, out_dims);

	md_slice(DIMS, flags, pos2, in_dims, out_data, in_data, CFL_SIZE);


	unmap_cfl(DIMS, out_dims, out_data);
	unmap_cfl(DIMS, in_dims, in_data);

	xfree(dims);
	xfree(poss);

	return 0;
}

