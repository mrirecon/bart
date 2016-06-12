/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "sense/recon.h"
#include "sense/optcom.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"




static const char usage_str[] = "<image> <kspace> <sens> <output>";
static const char help_str[] = "Recreate k-space from image and sensitivities.";




int main_fakeksp(int argc, char* argv[])
{
	bool rplksp = false;

	const struct opt_s opts[] = {

		OPT_SET('r', &rplksp, "replace measured samples with original values"),
	};

	cmdline(&argc, argv, 4, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);


	const int N = DIMS;
	long ksp_dims[N];
	long dims[N];
	long img_dims[N];

	complex float* kspace_data = load_cfl(argv[2], N, ksp_dims);
	complex float* sens_maps = load_cfl(argv[3], N, dims);
	complex float* image = load_cfl(argv[1], N, img_dims);
	

	for (int i = 0; i < 4; i++)
		if (ksp_dims[i] != dims[i])
			error("Dimensions of kspace and sensitivities do not match!\n");


	assert(1 == ksp_dims[MAPS_DIM]);
	assert(1 == img_dims[COIL_DIM]);
	assert(img_dims[MAPS_DIM] == dims[MAPS_DIM]);

	num_init();

	long dims1[N];

	md_select_dims(N, ~(COIL_FLAG|MAPS_FLAG), dims1, dims);

	long dims2[N];
	md_copy_dims(DIMS, dims2, img_dims);
	dims2[COIL_DIM] = dims[COIL_DIM];
	dims2[MAPS_DIM] = dims[MAPS_DIM];
	


#if 0
	float scaling = estimate_scaling(ksp_dims, NULL, kspace_data);
	printf("Scaling: %f\n", scaling);
	md_zsmul(N, ksp_dims, kspace_data, kspace_data, 1. / scaling);
#endif

	complex float* out = create_cfl(argv[4], N, ksp_dims);
	
	fftmod(N, ksp_dims, FFT_FLAGS, kspace_data, kspace_data);
	fftmod(N, dims, FFT_FLAGS, sens_maps, sens_maps);

	if (rplksp) {

		debug_printf(DP_INFO, "Replace kspace\n");
		replace_kspace(dims2, out, kspace_data, sens_maps, image); // this overwrites kspace_data (FIXME: think not!)

	} else {

		debug_printf(DP_INFO, "Simulate kspace\n");
		fake_kspace(dims2, out, sens_maps, image);
	}

#if 0
	md_zsmul(N, ksp_dims, out, out, scaling);
#endif
	fftmod(N, ksp_dims, FFT_FLAGS, out, out);

	unmap_cfl(N, ksp_dims, kspace_data);
	unmap_cfl(N, dims, sens_maps);
	unmap_cfl(N, img_dims, image);
	unmap_cfl(N, ksp_dims, out);

	exit(0);
}


