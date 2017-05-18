/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/casorati.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE	sizeof(complex float)
#endif

static const char usage_str[] = "dim1 kern1 dim2 kern2 ... dimn kernn <input> <output>";
static const char help_str[] = "Casorati matrix with kernel (kern1, ..., kernn) along dimensions (dim1, ..., dimn).\n";


int main_casorati(int argc, char* argv[])
{
	cmdline(&argc, argv, 4, 100, usage_str, help_str, 0, NULL);

	num_init();

	int count = argc - 3;
	
	assert((count > 0) && (count % 2 == 0));

	long idims[DIMS];
	long kdims[DIMS];
	long odims[2];

	complex float* idata = load_cfl(argv[argc - 2], DIMS, idims);

	md_copy_dims(DIMS, kdims, idims);

	for (int i = 0; i < count; i += 2) {

		unsigned int kdim = atoi(argv[i + 1]);
		unsigned int ksize = atoi(argv[i + 2]);

		assert(kdim < DIMS);
		assert(ksize >= 1);

		kdims[kdim] = ksize;
	}


	casorati_dims(DIMS, odims, kdims, idims);

	complex float* odata = create_cfl(argv[argc - 1], 2, odims);

	long istrs[DIMS];
	md_calc_strides(DIMS, istrs, idims, CFL_SIZE);

	casorati_matrix(DIMS, kdims, odims, odata, idims, istrs, idata);

	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(2, odims, odata);

	exit(0);
}
