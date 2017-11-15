/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <stdbool.h>

#include "linops/linop.h"

typedef struct nlop_data_s { TYPEID* TYPEID; } nlop_data_t;

typedef void (*nlop_fun_t)(const nlop_data_t* _data, complex float* dst, const complex float* src);
typedef void (*nlop_p_fun_t)(const nlop_data_t* _data, float lambda, complex float* dst, const complex float* src);
typedef void (*nlop_del_fun_t)(const nlop_data_t* _data);



struct operator_s;
struct linop_s;

struct nlop_s {

	const struct operator_s* op;
	const struct linop_s* derivative;
};



extern struct nlop_s* nlop_create(unsigned int ON, const long odims[__VLA(ON)], unsigned int IN, const long idims[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_fun_t deriv, nlop_fun_t adjoint, nlop_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t);

extern struct nlop_s* nlop_create2(unsigned int ON, const long odims[__VLA(ON)], const long ostr[__VLA(ON)],
				unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_fun_t deriv, nlop_fun_t adjoint, nlop_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t);


extern void nlop_free(const struct nlop_s* op);

extern nlop_data_t* nlop_get_data(struct nlop_s* op);

