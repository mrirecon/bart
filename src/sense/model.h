/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 

#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"

struct operator_s;
struct vec_ops;

extern struct linop_s* sense_init(const long max_dims[DIMS], unsigned int sens_flags, const complex float* sens, _Bool gpu);
extern struct linop_s* maps_create(const long max_dims[DIMS], 
			unsigned int sens_flags, const complex float* sens, bool gpu);
extern struct linop_s* maps2_create(const long coilim_dims[DIMS], const long maps_dims[DIMS], const long img_dims[DIMS], const complex float* maps, bool use_gpu);

extern void sense_free(const struct linop_s*);

extern void sense_forward(const struct linop_s* o, _Complex float* dst, const complex float* src);
extern void sense_adjoint(const struct linop_s* o, _Complex float* dst, const complex float* src);
extern void sense_normal(const struct linop_s* o, complex float* dst, const complex float* src);


#ifdef __cplusplus
}
#endif


