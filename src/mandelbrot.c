/* Copyright 2017-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/loop.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/stream.h"


static const char help_str[] = "Compute mandelbrot set.";



int main_mandelbrot(int argc, char* argv[argc])
{
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "output"),
	};

	int size = 512;
	int iter = 20;
	float zoom = .20; // 0.3
	float thresh = 4.;
	float offr = 0.0; // 0.4
	float offi = 0.0;
	bool save_iter = false;

	const struct opt_s opts[] = {

		OPT_PINT('s', &size, "size", "image size"),
		OPT_PINT('n', &iter, "#", "nr. of iterations"),
		OPT_FLOAT('t', &thresh, "t", "threshold for divergence"),
		OPT_FLOAT('z', &zoom, "z", "zoom"),
		OPT_FLOAT('r', &offr, "r", "offset real"),
		OPT_FLOAT('i', &offi, "i", "offset imag"),
		OPT_SET('I', &save_iter, "Save iterations"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	complex float off = offr + 1.i * offi;

	long dims[3] = { size, size, iter };

	complex float* o = create_async_cfl(out_file, MD_BIT(2), 3, dims);

	stream_t strm_o = stream_lookup(o);

	md_zfill(2, dims, o, iter);

	complex float* x = md_calloc(2, dims, CFL_SIZE);
	complex float* t = md_alloc(2, dims, CFL_SIZE);
	complex float* c = md_alloc(2, dims, CFL_SIZE);

	md_zgradient(2, dims, c, (complex float[2]){ 1., 1.i });
	md_zfill(2, dims, t, (size / 2.) * (1. + 1.i + off));
	md_zsub(2, dims, c, c, t);
	md_zsmul(2, dims, c, c, 1. / (zoom * size));

	complex float* occur = o;
	complex float* prev = o;
	long skip = md_calc_size(2, dims);

	for (int i = 0; i < iter; i++) {

		// iteration x -> x * x + c
		md_zmul(2, dims, x, x, x);
		md_zadd(2, dims, x, x, c);

		// track non-divergent points
		md_zabs(2, dims, t, x);
		md_slessequal(3, (long[3]){ 2, dims[0], dims[1] }, (float*)t, (float*)t, thresh);
		md_zreal(2, dims, t, t);
		md_zsub(2, dims, occur, prev, t);

		if (dims[2] > 1) {

			occur += skip;

			if (i != 0)
				prev += skip;
		}

		if (strm_o)
			stream_sync_slice(strm_o, 3, dims, MD_BIT(2), (long[3]){ [2] = i });
	}

	md_free(t);
	md_free(c);
	md_free(x);

	unmap_cfl(3, dims, o);

	return 0;
}

