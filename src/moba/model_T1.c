/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */


#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/debug.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/stack.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "noir/model.h"

#include "moba/T1fun.h"

#include "model_T1.h"




static struct mobamod T1_create_internal(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf,
				float scaling_M0, float scaling_R1s, const struct noir_model_conf_s* conf, float fov)
{
	long data_dims[DIMS];
	md_select_dims(DIMS, ~COEFF_FLAG, data_dims, dims);

	struct noir_s nlinv = noir_create3(data_dims, mask, psf, conf);
	struct mobamod ret;

	long map_dims[DIMS];
	long out_dims[DIMS];
	long in_dims[DIMS];
	long TI_dims[DIMS];

	md_select_dims(DIMS, conf->fft_flags|TIME_FLAG|TIME2_FLAG, map_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|TE_FLAG|TIME_FLAG|TIME2_FLAG, out_dims, dims);
	md_select_dims(DIMS, conf->fft_flags|COEFF_FLAG|TIME_FLAG|TIME2_FLAG, in_dims, dims);
	md_select_dims(DIMS, TE_FLAG|TIME_FLAG|TIME2_FLAG, TI_dims, dims);


#if 1
	// chain T1 model

	struct nlop_s* T1 = NULL;
	if(conf->noncart) { // overgridding with factor two

		long map_dims2[DIMS];
		long out_dims2[DIMS];
		long in_dims2[DIMS];

		md_copy_dims(DIMS, map_dims2, map_dims);
		md_copy_dims(DIMS, out_dims2, out_dims);
		md_copy_dims(DIMS, in_dims2, in_dims);

		long red_fov[3];

		for (int i = 0; i < 3; i++)
			red_fov[i] = (1 == map_dims[i]) ? 1 : (map_dims[i] * fov);

		if (1. != fov) {

			md_copy_dims(3, map_dims2, red_fov);
			md_copy_dims(3, out_dims2, red_fov);
			md_copy_dims(3, in_dims2, red_fov);
		}

		T1 = nlop_T1_create(DIMS, map_dims2, out_dims2, in_dims2, TI_dims, TI, scaling_M0, scaling_R1s);

		T1 = nlop_chain_FF(T1, nlop_from_linop_F(linop_resize_center_create(DIMS, out_dims, out_dims2)));
		T1 = nlop_chain_FF(nlop_from_linop_F(linop_resize_center_create(DIMS, in_dims2, in_dims)), T1);

	} else
		T1 = nlop_T1_create(DIMS, map_dims, out_dims, in_dims, TI_dims, TI, scaling_M0, scaling_R1s);

	debug_printf(DP_INFO, "T1 Model created:\n Model ");
	nlop_debug(DP_INFO, T1);

	debug_printf(DP_INFO, "NLINV ");
	nlop_debug(DP_INFO, nlinv.nlop);

	const struct nlop_s* b = nlinv.nlop;
	const struct nlop_s* c = nlop_chain2_FF(T1, 0, b, 0);

	nlinv.nlop = nlop_permute_inputs_F(c, 2, (const int[2]){ 1, 0 });
#endif
	ret.nlop = nlinv.nlop;
	ret.linop = nlinv.linop;

	return ret;
}


struct mobamod T1_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf, 
			float scaling_M0, float scaling_R1s, const struct noir_model_conf_s* conf, float fov)
{
	unsigned long bat_flags = TIME_FLAG | TIME2_FLAG;
	int bat_idx = TIME_DIM;
	
	if(1 < dims[TIME2_DIM]) {

		if (   ((0 == MD_IS_SET(conf->ptrn_flags, TIME_DIM)) != (0 == MD_IS_SET(conf->ptrn_flags, TIME2_DIM)))
		    || ((0 == MD_IS_SET(conf->cnstcoil_flags, TIME_DIM)) != (0 == MD_IS_SET(conf->cnstcoil_flags, TIME2_DIM))) ) {

			bat_flags = MD_CLEAR(bat_flags, TIME_DIM);
			bat_idx = TIME2_DIM;
	    	}  
	}

	long bat_dims[DIMS];
	long dims_slc[DIMS];

	md_select_dims(DIMS,  bat_flags, bat_dims, dims);
	md_select_dims(DIMS, ~bat_flags, dims_slc, dims);

	long psf_dims[DIMS];
	md_select_dims(DIMS, conf->ptrn_flags & ~COEFF_FLAG, psf_dims, dims);

	long psf_dims_slc[DIMS];
	md_select_dims(DIMS, ~bat_flags, psf_dims_slc, psf_dims);

	long TI_dims[DIMS];
	md_select_dims(DIMS, TE_FLAG|TIME_FLAG|TIME2_FLAG, TI_dims, dims);

	long TI_dims_slc[DIMS];
	md_select_dims(DIMS, ~bat_flags, TI_dims_slc, TI_dims);

	int N = md_calc_size(DIMS, bat_dims);
	
	const struct linop_s* lop = NULL;
	const struct nlop_s* nlops[N];

	long pos[DIMS];
	md_singleton_strides(DIMS, pos);

	int i = 0;

	do {
		complex float* psf_tmp = md_alloc(DIMS, psf_dims_slc, CFL_SIZE);
		md_slice(DIMS, bat_flags, pos, psf_dims, psf_tmp, psf, CFL_SIZE);

		complex float* TI_tmp = md_alloc(DIMS, psf_dims_slc, CFL_SIZE);
		md_slice(DIMS, bat_flags, pos, TI_dims, TI_tmp, TI, CFL_SIZE);

		struct mobamod T1 = T1_create_internal(dims_slc, mask, TI_tmp, psf_tmp, scaling_M0, scaling_R1s, conf, fov);	

		nlops[i++] = T1.nlop;	
		if (NULL == lop)
			lop = linop_clone(T1.linop);
		else {
			if (!MD_IS_SET(conf->cnstcoil_flags, bat_idx))
				lop = linop_stack_FF(bat_idx, bat_idx, lop, linop_clone(T1.linop));
		}

		md_free(TI_tmp);
		md_free(psf_tmp);

	} while (md_next(DIMS, bat_dims, bat_flags, pos));

	struct mobamod result = {

		.nlop = (1 == N) ? (struct nlop_s*)nlops[0] : (struct nlop_s*)nlop_stack_container_create_F(N, nlops, 2, (int [2]){ bat_idx, (MD_IS_SET(conf->cnstcoil_flags, bat_idx) ? -1 : bat_idx) }, 1, (int[1]){ bat_idx }),
		.linop = lop,
	};

	result.nlop = nlop_flatten_F(result.nlop);

	return result;
}
