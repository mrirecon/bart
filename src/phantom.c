/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "simu/phantom.h"



static const char* usage_str = "<output>";
static const char* help_str = "Image and k-space domain phantoms.";




int main_phantom(int argc, char* argv[])
{
	bool kspace = false;
	int sens = 0;
	int osens = -1;
	int xdim = -1;
	bool out_sens = false;
	bool tecirc = false;
	bool circ = false;
	char* traj = NULL;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 1;
	dims[1] = 128;
	dims[2] = 128;


	const struct opt_s opts[] = {

		{ 's', true, opt_int, &sens, " nc\tnc sensitivities" },
		{ 'S', true, opt_int, &osens, NULL },
		{ 'k', false, opt_set, &kspace, "\tk-space" },
		{ 't', true, opt_string, &traj, NULL },
		{ 'c', false, opt_set, &circ, NULL },
		{ 'm', false, opt_set, &tecirc, NULL },
		{ 'x', true, opt_int, &xdim, NULL },
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (tecirc) {

		circ = true;
		dims[TE_DIM] = 32;
	}

	if (-1 != osens) {

		out_sens = true;
		sens = osens;
	}

	if (-1 != xdim)
		dims[1] = dims[2] = xdim;


	long sdims[DIMS];
	complex float* samples = NULL;

	if (NULL != traj) {

		samples = load_cfl(traj, DIMS, sdims);

		dims[0] = 1;
		dims[1] = sdims[1];
		dims[2] = sdims[2];
	}


	if (sens)
		dims[3] = sens;

	complex float* out = create_cfl(argv[1], DIMS, dims);

	if (out_sens) {

		assert(NULL == traj);
		assert(!kspace);

		calc_sens(dims, out);

	} else
	if (circ) {

		assert(NULL == traj);

		if (1 < dims[TE_DIM])
			calc_moving_circ(dims, out, kspace);
		else
			calc_circ(dims, out, kspace);
//		calc_ring(dims, out, kspace);

	} else {

		//assert(1 == dims[COIL_DIM]);

		if (NULL == samples) {

			calc_phantom(dims, out, kspace);

		} else {

			dims[0] = 3;
			calc_phantom_noncart(dims, out, samples);
			dims[0] = 1;
		}
	}

	if (NULL != traj)
		free(traj);

	if (NULL != samples)
		unmap_cfl(3, sdims, samples);

	unmap_cfl(DIMS, dims, out);
	exit(0);
}


