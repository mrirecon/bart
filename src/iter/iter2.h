/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __ITER2_H
#define __ITER2_H

struct linop_s;
struct operator_s;
struct operator_p_s;

#ifndef ITER_CONF_S
#define ITER_CONF_S
typedef struct iter_conf_s { int:0; } iter_conf;
#endif

#ifndef __VLA
#ifdef __cplusplus
#define __VLA(x)
#else
#define __VLA(x) static x
#endif
#endif

typedef void (italgo_fun2_f)(iter_conf* conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s* prox_ops[__VLA(D)],
		const struct linop_s* ops[__VLA(D)],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*));

typedef italgo_fun2_f* italgo_fun2_t;

italgo_fun2_f iter2_conjgrad;
italgo_fun2_f iter2_ist;
italgo_fun2_f iter2_fista;
italgo_fun2_f iter2_admm;
italgo_fun2_f iter2_pocs;


// use with iter_call_s from iter.h as _conf
italgo_fun2_f iter2_call_iter;


struct iter2_call_s {

	iter_conf base;

	italgo_fun2_t fun;
	iter_conf* _conf;
};


#endif

