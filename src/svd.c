/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013, Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <getopt.h>
#include <stdio.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/lapack.h"

#include "misc/misc.h"
#include "misc/mmio.h"



static const char* usage_str = "[-e] <input> <U> <S> <VH>";
static const char* help_str = 	"Compute singular-value-decomposition (SVD).\n";

static void usage(FILE* fp, const char* name)
{
	fprintf(fp, "Usage %s: %s\n", name, usage_str);
}

static void help(void)
{
	printf("\n%s", help_str);
}


int main_svd(int argc, char* argv[])
{
	bool econ = false;

	int c;
	while (-1 != (c = getopt(argc, argv, "eh"))) {

		switch (c) {

		case 'e':
			econ = true;
			break;

		case 'h':
			usage(stdout, argv[0]);
			help();
			exit(0);

		default:
			usage(stderr, argv[0]);
			exit(1);
		}
	}

	if (4 != argc - optind) {

		usage(stderr, argv[0]);
		exit(1);
	}

	int N = 2;
	long dims[N];

	complex float* in = load_cfl(argv[optind + 0], N, dims);

	long dimsU[2] = { dims[0], econ ? MIN(dims[0], dims[1]) : dims[0] };
	long dimsS[2] = { MIN(dims[0], dims[1]), 1 };
	long dimsVH[2] = { econ ? MIN(dims[0], dims[1]) : dims[1], dims[1] };

	complex float* U = create_cfl(argv[optind + 1], N, dimsU);
	complex float* S = create_cfl(argv[optind + 2], N, dimsS);
	complex float* VH = create_cfl(argv[optind + 3], N, dimsVH);

	float* SF = md_alloc(2, dimsS, FL_SIZE);

	(econ ? lapack_svd_econ : lapack_svd)(dims[0], dims[1],
			MD_CAST_ARRAY2(complex float, 2, dimsU, U, 0, 1),
			MD_CAST_ARRAY2(complex float, 2, dimsVH, VH, 0, 1),
			SF, MD_CAST_ARRAY2(complex float, 2, dims, in, 0, 1));

	for (int i = 0 ; i < dimsS[0]; i++)
		S[i] = SF[i];

	md_free(SF);


	unmap_cfl(N, dims, in);
	unmap_cfl(N, dimsU, U);
	unmap_cfl(N, dimsS, S);
	unmap_cfl(N, dimsVH, VH);
	exit(0);
}


