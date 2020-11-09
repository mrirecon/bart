/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2019-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#ifndef __RECON_MECO_H
#define __RECON_MECO_H

#include "iter/iter.h"
#include "iter/iter2.h"

#include "noir/recon.h"

struct moba_conf;

void init_meco_maps(const long maps_dims[DIMS], complex float* maps, unsigned int sel_model);

void meco_recon(struct moba_conf* moba_conf, 
		unsigned int sel_model, bool real_pd, 
		float* scale_fB0, bool warmstart, bool out_origin_maps, 
		const long maps_dims[DIMS], complex float* maps, 
		const long sens_dims[DIMS], complex float* sens, 
		const long init_dims[DIMS], complex float* init, 
		const complex float* mask, 
		const complex float* TE, 
		const long P_dims[DIMS], complex float* P, 
		const long Y_dims[DIMS], complex float* Y);

#endif
