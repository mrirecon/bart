/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Nick Scholand
 */

#include <stdbool.h>
#include <complex.h>
#include <stdint.h>

#include "num/flpmath.h"

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noncart/traj.h"

static const char help_str[] = "Generate file with RAGA indices for given approximated tiny golden ratio angle/raga increment and full frame spokes.";


int main_raga(int argc, char* argv[argc])
{
	const char* out_file= NULL;
	int Y;

	struct arg_s args[] = {

		ARG_INT(true, &Y, "spokes"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	int raga_inc = 0;
	int tiny_gold = 0;
	bool double_base = true;

	const struct opt_s opts[] = {

		OPTL_PINT('s', "tiny-angle", &tiny_gold, "# Tiny GA", "tiny (small) golden ratio angle"),
		OPTL_PINT('r', "raga-inc", &raga_inc, "d", "Increment of RAGA Sampling"),
		OPTL_CLEAR(0, "no-double-base", &double_base, "Define GA over Pi base instead of 2Pi."),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(0 < Y);
	assert(raga_inc < Y);

	// Recover tiny golden angle from raga_inc if it was not passed

	if ((0 == tiny_gold) && (0 == raga_inc))
		tiny_gold = 1;

	assert((0 == Y % 2) || double_base);

	if ((0 == tiny_gold) && (0 != raga_inc))
		tiny_gold = recover_gen_fib_ind(Y / (double_base ? 1 : 2), raga_inc);

	if (-1 == tiny_gold)
		error("Could not recover index of approximated golden ratio angle!\n");

	if ((0 < tiny_gold) && (0 != raga_inc))
		assert(tiny_gold == recover_gen_fib_ind(Y / (double_base ? 1 : 2), raga_inc));

	debug_printf(DP_INFO, "Golden Ratio Index is set to:\t%d\n", tiny_gold);

	assert(0 < tiny_gold);

	// Generate index file

	long dims[DIMS] = { [0 ... DIMS - 1] = 1  };
	dims[PHS2_DIM] = Y;

	complex float* indices = create_cfl(out_file, DIMS, dims);
	md_clear(DIMS, dims, indices, CFL_SIZE);

	int p = 0;
	long pos[DIMS] = { };

	do {
		int j = pos[PHS2_DIM];

		indices[p] = (j * raga_increment(Y  / (double_base ? 1 : 2), tiny_gold)) % Y;

		p++;

	} while (md_next(DIMS, dims, ~1UL, pos));

	assert(p == Y);

	unmap_cfl(DIMS, dims, indices);

	return 0;
}



