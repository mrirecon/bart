/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Frank Ong <frankong@berkeley.edu>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "noncart/nufft.h"


static const char usage_str[] = "<traj>";
static const char help_str[] = "Estimate image dimension from non-Cartesian trajectory.\n"
			"Assume trajectory scaled to -DIM/2 to DIM/2 (ie dk=1/FOV=1)\n";



int main_estdims(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 1, usage_str, help_str);

	num_init();

	int N = 16;

	long traj_dims[N];
	
	complex float* traj = load_cfl(argv[1], N, traj_dims);

	long im_dims[N];
	
	estimate_im_dims(N, im_dims, traj_dims, traj);

	printf("%ld %ld %ld\n", im_dims[0], im_dims[1], im_dims[2]);
	
	unmap_cfl(N, traj_dims, traj);
	exit(0);
}

