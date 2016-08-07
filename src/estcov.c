/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2016 Siddharth Iyer <sid8795@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "misc/mmio.h"
#include "misc/opts.h"

#include "calib/estcov.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<kspace> <cov mat>.\n";
static const char help_str[] = "Estimate the noise covariance matrix across channel from the edge of kspace data.\n"
			       "The width of the edge along a dimension is determined by p * (dim size).\n"; 


int main_estcov(int argc, char* argv[])
{
	float p = 0.3;
	const struct opt_s opts[1] = {
		OPT_FLOAT('p', &p, "ratio", "0 <= p <= 1. Ratio that determines edges based on kspace size."),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, 1, opts);

	assert(p >=0 && p <= 1);

	long N = DIMS;
	long kspace_dims[N];
	complex float* kspace = load_cfl(argv[1], N, kspace_dims);

	long nc = kspace_dims[3];

	long cov_dims[2] = {nc, nc};
	complex float* cov = create_cfl(argv[2], 2, cov_dims);

	estcov(cov_dims, cov, p, N, kspace_dims, kspace);

	unmap_cfl(2, cov_dims, cov);
	exit(0);
}
