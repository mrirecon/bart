/* Copyright 2025. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/loop.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"


static const char help_str[] = "Compute histogram from -0.5 to 0.5.";



int main_hist(int argc, char* argv[argc])
{
	const char* in_file;
	const char* out_file;

	unsigned long flags;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "bitmask"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	int size = 100;
	bool cmplx = false;

	const struct opt_s opts[] = {

		OPT_SET('c', &cmplx, "complex"),
		OPT_PINT('s', &size, "size", "number of bins"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	enum { D = 16 };

	long idims[D];
	const complex float *in = load_cfl(in_file, D, idims);

	long bdims[D];
	md_select_dims(D, flags, bdims, idims);

	int N = md_calc_size(D, bdims);
	int N2 = N * (cmplx ? 2 : 1);

	assert(N2 <= D);

	complex float *tmp = md_alloc(D, bdims, sizeof *tmp);

	long odims[D];
	md_singleton_dims(D, odims);

	for (int i = 0; i < N2; i++)
		odims[i] = size;

	complex float *out = create_cfl(out_file, D, odims);

	md_zfill(D, odims, out, 0.);

	long ostrs[D];
	md_calc_strides(D, ostrs, odims, sizeof *out);

	long pos[D] = { };

	do {
		md_copy_block(D, pos, bdims, tmp, idims, in, sizeof *in);

		long opos[D];

		for (int i = 0; i < N; i++) {

			int d = cmplx ? 2 : 1;

			opos[i * d + 0] = ceilf(size * (crealf(tmp[i]) + 0.5));

			if (cmplx)
				opos[i * d + 1] = ceilf(size * (cimagf(tmp[i]) + 0.5));
		}

		bool ok = true;

		for (int i = 0; i < N2; i++)
			ok &= ((0 <= opos[i]) && (opos[i] < size));

		if (ok)
			MD_ACCESS(D, ostrs, opos, out)++;

	} while (md_next(D, idims, ~flags, pos));

	md_free(tmp);

	unmap_cfl(D, odims, out);
	unmap_cfl(D, idims, in);

	return 0;
}

