/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 


#include "misc/mri.h"

struct noir_data;

extern void noir_fun(struct noir_data*, complex float* dst, const complex float* src);
extern void noir_der(struct noir_data*, complex float* dst, const complex float* src);
extern void noir_adj(struct noir_data*, complex float* dst, const complex float* src);

extern void noir_forw_coils(struct noir_data* data, complex float* dst, const complex float* src);
extern void noir_back_coils(struct noir_data* data, complex float* dst, const complex float* src);

struct noir_model_conf_s {

	unsigned int fft_flags;
	_Bool rvc;
	_Bool use_gpu;
	_Bool noncart;
	_Bool pattern_for_each_coil;
	float a;
	float b;
};

extern struct noir_model_conf_s noir_model_conf_defaults;

extern struct noir_data* noir_init(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf);
extern void noir_free(struct noir_data* data);

extern void noir_orthogonalize(struct noir_data*, complex float* dst, const complex float* src);



