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

#include "moba/meco.h"


struct moba_conf;
enum fat_spec;

void init_meco_maps(const long maps_dims[DIMS], complex float* maps, enum meco_model sel_model);

void meco_recon(const struct moba_conf* moba_conf,
		enum meco_model sel_model, bool real_pd, enum fat_spec fat_spec,
		const float* scale_fB0, bool warmstart, bool out_origin_maps,
		const long maps_dims[DIMS], complex float* maps,
		const long sens_dims[DIMS], complex float* sens,
		const long init_dims[DIMS], const complex float* init,
		const complex float* mask,
		const complex float* TE,
		const long P_dims[DIMS], const complex float* P,
		const long Y_dims[DIMS], const complex float* Y);

#endif
