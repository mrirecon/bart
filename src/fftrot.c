/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "num/init.h"
#include "num/fft.h"
#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char usage_str[] = "dim1 dim2 theta <input> <output>";
static const char help_str[] = "Performs a rotation using Fourier transform (FFT) along selected dimensions.";



int main_fftrot(int argc, char* argv[])
{
	const struct opt_s opts[] = {

	};

	cmdline(&argc, argv, 5, 5, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;
	long dims[N];

	complex float* idata = load_cfl(argv[4], N, dims);
	complex float* odata = create_cfl(argv[5], N, dims);

	int dim1 = atoi(argv[1]);
	int dim2 = atoi(argv[2]);

	assert(dim1 != dim2);

	if (dim1 > dim2) {

		int tmp = dim1;
		dim1 = dim2;
		dim2 = tmp;
	}

	float theta = atof(argv[3]);

	float alpha = -tanf(theta / 2.);
	float beta = sinf(theta);

	long flags = (1 << dim1) | (1 << dim2);

	long phdims[N];
	md_select_dims(N, flags, phdims, dims);

	complex float* phx = md_alloc(N, phdims, CFL_SIZE);
	complex float* phy = md_alloc(N, phdims, CFL_SIZE);

	int X = phdims[dim1];
	int Y = phdims[dim2];

	for (int i = 0; i < X; i++) {
		for (int j = 0; j < Y; j++) {

			phx[i + X * j] = cexpf(+2.i * M_PI * ((j - Y / 2.) * alpha * (i - X / 2.) / (float)(X)));
			phy[i + X * j] = cexpf(-2.i * M_PI * ((j - Y / 2.) * beta * (i - X / 2.) / (float)(Y)));
		}
	}

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	long phstrs[N];
	md_calc_strides(N, phstrs, phdims, CFL_SIZE);

	fftuc(N, dims, (1u << dim1), odata, idata);

	md_zmul2(N, dims, strs, odata, strs, odata, phstrs, phx);

	ifftuc(N, dims, flags, odata, odata);

	md_zmul2(N, dims, strs, odata, strs, odata, phstrs, phy);

	fftuc(N, dims, flags, odata, odata);

	md_zmul2(N, dims, strs, odata, strs, odata, phstrs, phx);

	ifftuc(N, dims, (1u << dim1), odata, odata);

	md_free(phx);
	md_free(phy);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);

	return 0;
}


