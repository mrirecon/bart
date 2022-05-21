/* Copyright 2022. Institute of Medical Engineering. Graz University of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


#include <stdbool.h>
#include <assert.h>

#include "num/multind.h"

#include "moba/moba.h"
#include "moba/recon_T1.h"
#include "moba/recon_T2.h"
#include "moba/recon_meco.h"

#include "recon.h"

void moba_recon(const struct moba_conf* conf, const long dims[DIMS], complex float* img, complex float* sens, const complex float* pattern, const complex float* mask, const complex float* TI, const complex float* kspace_data, const complex float* init)
{
	switch (conf->mode) {

	case MDB_T1:

		assert(NULL == init);
		T1_recon(conf, dims, img, sens, pattern, mask, TI, kspace_data, conf->use_gpu);
		break;

	case MDB_T2:

		assert(NULL == init);
		T2_recon(conf, dims, img, sens, pattern, mask, TI, kspace_data, conf->use_gpu);
		break;

	case MDB_MGRE:

		;

		long imgs_dims[DIMS];
		long coil_dims[DIMS];
		long data_dims[DIMS];
		long pat_dims[DIMS];

		unsigned int fft_flags = FFT_FLAGS;

		if (conf->sms)
			fft_flags |= SLICE_FLAG;

		md_select_dims(DIMS, fft_flags|MAPS_FLAG|CSHIFT_FLAG|COEFF_FLAG|TIME2_FLAG, imgs_dims, dims);
		md_select_dims(DIMS, fft_flags|COIL_FLAG|MAPS_FLAG|TIME2_FLAG, coil_dims, dims);
		md_select_dims(DIMS, fft_flags|COIL_FLAG|TE_FLAG|MAPS_FLAG|TIME2_FLAG, data_dims, dims);
		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, data_dims);


		meco_recon(conf, conf->mgre_model, false, conf->fat_spec, conf->scale_fB0, true, conf->out_origin_maps, imgs_dims, img, coil_dims, sens, imgs_dims, init, mask, TI, pat_dims, pattern, data_dims, kspace_data);
		break;

	default:
		assert(0);
	}
}
		
