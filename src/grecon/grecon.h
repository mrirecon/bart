/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */


#ifndef __GRECON_H
#define __GRECON_H 1
 

#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"


struct ecalib_conf;

enum algo { SENSE, POCS, NOIR };

struct grecon_conf {

	enum algo algo;
	const struct ecalib_conf* calib;
	struct sense_conf* sense_conf;
	bool rplksp;
	bool ksp;
	bool l1wav;
	bool randshift;
	int maxiter;
	float step;
	float lambda;
};

extern void grecon(struct grecon_conf* param,  const long dims1[DIMS], _Complex float* out1, 
	const long sens1_dims[DIMS], _Complex float* cov1, 
	const long w1_dims[DIMS], const _Complex float* weights,
	_Complex float* kspace1, bool usegpu);

extern void rgrecon(struct grecon_conf* conf, const long dims[DIMS], _Complex float* image,
			const long sens_dims[DIMS], const _Complex float* sens_maps,
			const long pat1_dims[DIMS], const _Complex float* weights, 
			const _Complex float* kspace_data, bool usegpu);

extern void rgrecon2(struct grecon_conf* conf, const long dims[DIMS], 
			const long img_strs[DIMS], _Complex float* image,
			const long sens_dims[DIMS], const _Complex float* sens_maps,
			const long pat1_dims[DIMS], const _Complex float* weights, 
			const long ksp_strs[DIMS], const _Complex float* kspace, bool usegpu);



#ifdef __cplusplus
}
#endif


#endif // __GRECON_H

