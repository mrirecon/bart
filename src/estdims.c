/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Frank Ong <frankong@berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "noncart/nufft.h"


#define FFT_DIMS (MD_BIT(0)|MD_BIT(1)|MD_BIT(2))

static const char help_str[] = "Estimate image dimension from non-Cartesian trajectory.\n"
			"Assume trajectory scaled to -DIM/2 to DIM/2 (ie dk=1/FOV=1)";



int main_estdims(int argc, char* argv[argc])
{
	const char* traj_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &traj_file, "traj"),
	};

	const struct opt_s opts[] = {};
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = 16;

	long traj_dims[N];
	
	complex float* traj = load_cfl(traj_file, N, traj_dims);

	long im_dims[N];
	
	estimate_im_dims(N, FFT_DIMS, im_dims, traj_dims, traj);

	bart_printf("%ld %ld %ld\n", im_dims[0], im_dims[1], im_dims[2]);
	
	unmap_cfl(N, traj_dims, traj);
	return 0;
}

