/* Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2020-2022 Martin Uecker <uecker@tugraz.at>
 */

#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/mri.h"



static complex float* noncart_shift(const float shift[3], const long tdims[DIMS], const complex float* tdata)
{
	long odims[DIMS];
	md_select_dims(DIMS, ~1u, odims, tdims);

	complex float* odata = md_alloc(DIMS, odims, CFL_SIZE);

	complex float cshift[3] = { shift[0], shift[1], shift[2] };

	long shift_dims[DIMS] = { 3, [1 ... DIMS - 1] = 1 };

	md_ztenmul2(DIMS, tdims, MD_STRIDES(DIMS, odims, CFL_SIZE), odata,
				MD_STRIDES(DIMS, tdims, CFL_SIZE), tdata,
				MD_STRIDES(DIMS, shift_dims, CFL_SIZE), cshift);


	md_zsmul(DIMS, odims, odata, odata, +2.i * M_PI);
	md_zexp(DIMS, odims, odata, odata);

	return odata;
}


static const char help_str[] = "Shifts FOV.";


int main_fovshift(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	float shift[3] = { 0., 0., 0. };
	const char* traj_file = NULL;

	const struct opt_s opts[] = {

		OPT_INFILE('t', &traj_file, "file", "k-space trajectory"),
		OPT_FLVEC3('s', &shift, "X:Y:Z", "FOV shift"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	long idims[DIMS];
	complex float* idata = load_cfl(in_file, DIMS, idims);

	long pdims[DIMS];
	complex float* phase;

	if (NULL != traj_file) {

		long tdims[DIMS];
		complex float* tdata = load_cfl(traj_file, DIMS, tdims);

		md_select_dims(DIMS, ~1u, pdims, tdims);

		phase = noncart_shift(shift, tdims, tdata);

		unmap_cfl(DIMS, tdims, tdata);

	} else {

		md_select_dims(DIMS, FFT_FLAGS, pdims, idims);

		phase = md_alloc(DIMS, pdims, CFL_SIZE);

		for (int i = 0; i < 3; i++)
			shift[i] *= idims[i];

		linear_phase(3, pdims, shift, phase);
	}

	complex float* odata = create_cfl(out_file, DIMS, idims);

	md_zmul2(DIMS, idims, MD_STRIDES(DIMS, idims, CFL_SIZE), odata,
			MD_STRIDES(DIMS, idims, CFL_SIZE), idata,
			MD_STRIDES(DIMS, pdims, CFL_SIZE), phase);

	md_free(phase);

	unmap_cfl(DIMS, idims, idata);
	unmap_cfl(DIMS, idims, odata);

	return 0;
}
