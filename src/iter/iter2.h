/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __ITER2_H
#define __ITER2_H

// for prox_fun_t
#include "iter/admm.h"

struct linop_s;
struct operator_s;
struct operator_p_s;

typedef void (*italgo_fun2_t)(void* conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*));



extern void iter2_conjgrad(void* conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*));

extern void iter2_ist(void* conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*));

extern void iter2_fista(void* conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*));

extern void iter2_admm(void* conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*));


void iter2_pocs(void* _conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s** prox_ops,
		const struct linop_s** ops,
		const struct operator_s* xupdate_op,
		long size, float* image, const float* image_adj,
		const float* image_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*));


#endif

