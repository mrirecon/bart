/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/loop.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"


static const char usage_str[] = "output";
static const char help_str[] = "Compute mandelbrot set.\n";



int main_mandelbrot(int argc, char* argv[])
{
	unsigned int size = 512;
	unsigned int iter = 20;
	float zoom = .20; // 0.3
	float thresh = 4.;
	float offr = 0.0; // 0.4
	float offi = 0.0;

	const struct opt_s opts[] = {

		OPT_UINT('s', &size, "size", "image size"),
		OPT_UINT('n', &iter, "#", "nr. of iterations"),
		OPT_FLOAT('t', &thresh, "t", "threshold for divergence"),
		OPT_FLOAT('z', &zoom, "z", "zoom"),
		OPT_FLOAT('r', &offr, "r", "offset real"),
		OPT_FLOAT('i', &offi, "i", "offset imag"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	
	complex float off = offr + 1.i * offi;

	long dims[2] = { size, size };

	complex float* o = create_cfl(argv[1], 2, dims);
	md_zfill(2, dims, o, iter);

	complex float* x = md_calloc(2, dims, CFL_SIZE);

	complex float* t = md_alloc(2, dims, CFL_SIZE);

	complex float* c = md_alloc(2, dims, CFL_SIZE);
	md_zgradient(2, dims, c, (complex float[2]){ 1., 1.i });
	md_zfill(2, dims, t, (size / 2.) * (1. + 1.i + off));
	md_zsub(2, dims, c, c, t);
	md_zsmul(2, dims, c, c, 1. / (zoom * size));

	for (unsigned int i = 0; i < iter; i++) {

		// iteration x -> x * x + c
		md_zmul(2, dims, x, x, x);
		md_zadd(2, dims, x, x, c);
	
		// track non-divergent points
		md_zabs(2, dims, t, x);
		md_slessequal(3, (long[3]){ 2, dims[0], dims[1] }, (float*)t, (float*)t, thresh);
		md_zreal(2, dims, t, t);
		md_zsub(2, dims, o, o, t);
	}

	md_free(t);
	md_free(c);
	md_free(x);

	unmap_cfl(2, dims, o);
	exit(0);
}


