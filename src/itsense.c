/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * 
 * Basic iterative sense reconstruction
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"

#include "iter/iter.h"

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/types.h"


struct sense_data {

	INTERFACE(operator_data_t);

	long sens_dims[DIMS];
	long sens_strs[DIMS];

	long imgs_dims[DIMS];
	long imgs_strs[DIMS];

	long data_dims[DIMS];
	long data_strs[DIMS];

	long mask_dims[DIMS];
	long mask_strs[DIMS];

	const complex float* sens;
	const complex float* pattern;
	complex float* tmp;

	float alpha;
}; 

DEF_TYPEID(sense_data);


static void sense_forward(const struct sense_data* data, complex float* out, const complex float* imgs)
{
	md_clear(DIMS, data->data_dims, out, CFL_SIZE);
	md_zfmac2(DIMS, data->sens_dims, data->data_strs, out, data->sens_strs, data->sens, data->imgs_strs, imgs); 

	fftc(DIMS, data->data_dims, FFT_FLAGS, out, out);
	fftscale(DIMS, data->data_dims, FFT_FLAGS, out, out);

	md_zmul2(DIMS, data->data_dims, data->data_strs, out, data->data_strs, out, data->mask_strs, data->pattern);
}


static void sense_adjoint(const struct sense_data* data, complex float* imgs, const complex float* out)
{
	md_zmulc2(DIMS, data->data_dims, data->data_strs, data->tmp, data->data_strs, out, data->mask_strs, data->pattern);

	ifftc(DIMS, data->data_dims, FFT_FLAGS, data->tmp, data->tmp);
	fftscale(DIMS, data->data_dims, FFT_FLAGS, data->tmp, data->tmp);

	md_clear(DIMS, data->imgs_dims, imgs, CFL_SIZE);
	md_zfmacc2(DIMS, data->sens_dims, data->imgs_strs, imgs, data->data_strs, data->tmp, data->sens_strs, data->sens);
}


static void sense_normal(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const struct sense_data* data = CAST_DOWN(sense_data, _data);

	assert(2 == N);
	float* out = args[0];
	const float* in = args[1];

	sense_forward(data, data->tmp, (const complex float*)in);
	sense_adjoint(data, (complex float*)out, data->tmp);
}





static void sense_reco(struct sense_data* data, complex float* imgs, const complex float* kspace)
{
	complex float* adj = md_alloc(DIMS, data->imgs_dims, CFL_SIZE);

	md_clear(DIMS, data->imgs_dims, imgs, CFL_SIZE);

	sense_adjoint(data, adj, kspace);

	long size = md_calc_size(DIMS, data->imgs_dims);

	const struct operator_s* op = operator_create(DIMS, data->imgs_dims, DIMS, data->imgs_dims,
		CAST_UP(data), sense_normal, NULL);

	struct iter_conjgrad_conf conf = iter_conjgrad_defaults;
	conf.maxiter = 100;
	conf.l2lambda = data->alpha;
	conf.tol = 1.E-3;

	iter_conjgrad(CAST_UP(&conf), op, NULL, size, (float*)imgs, (const float*)adj, NULL);

	operator_free(op);

	md_free(adj);
}



static bool check_dimensions(struct sense_data* data)
{
	bool ok = true;

	for (int i = 0; i < 3; i++) {

		ok &= (data->mask_dims[i] == data->sens_dims[i]);
		ok &= (data->data_dims[i] == data->sens_dims[i]);
		ok &= (data->imgs_dims[i] == data->sens_dims[i]);
	}

	ok &= (data->data_dims[COIL_DIM] == data->sens_dims[COIL_DIM]);
	ok &= (data->imgs_dims[MAPS_DIM] == data->sens_dims[MAPS_DIM]);

	ok &= (1 == data->data_dims[MAPS_DIM]);
	ok &= (1 == data->mask_dims[COIL_DIM]);
	ok &= (1 == data->mask_dims[MAPS_DIM]);
	ok &= (1 == data->imgs_dims[COIL_DIM]);

	return ok;	
}



static const char usage_str[] = "alpha <sensitivities> <kspace> <pattern> <image>";
static const char help_str[] = "A simplified implementation of iterative sense reconstruction\n"
				"with l2-regularization.\n";



int main_itsense(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 5, usage_str, help_str);

	struct sense_data data;
	SET_TYPEID(sense_data, &data);

	data.alpha = atof(argv[1]);

	complex float* kspace = load_cfl(argv[3], DIMS, data.data_dims);

	data.sens = load_cfl(argv[2], DIMS, data.sens_dims);
	data.pattern = load_cfl(argv[4], DIMS, data.mask_dims);

	// 1 2 4 8
	md_select_dims(DIMS, ~COIL_FLAG, data.imgs_dims, data.sens_dims);

	assert(check_dimensions(&data));

	complex float* image = create_cfl(argv[5], DIMS, data.imgs_dims);

	md_calc_strides(DIMS, data.sens_strs, data.sens_dims, CFL_SIZE);
	md_calc_strides(DIMS, data.imgs_strs, data.imgs_dims, CFL_SIZE);
	md_calc_strides(DIMS, data.data_strs, data.data_dims, CFL_SIZE);
	md_calc_strides(DIMS, data.mask_strs, data.mask_dims, CFL_SIZE);

	data.tmp = md_alloc(DIMS, data.data_dims, CFL_SIZE);

	num_init();

	sense_reco(&data, image, kspace);

	unmap_cfl(DIMS, data.imgs_dims, image);
	unmap_cfl(DIMS, data.mask_dims, data.pattern);
	unmap_cfl(DIMS, data.sens_dims, data.sens);
	unmap_cfl(DIMS, data.data_dims, data.sens);
	md_free(data.tmp);

	exit(0);
}


