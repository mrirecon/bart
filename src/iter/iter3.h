/* Copyright 2013-2014. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/types.h"
typedef struct iter3_conf_s { TYPEID* TYPEID; } iter3_conf;


typedef void iter3_fun_f(iter3_conf* _conf,
		void (*frw)(void* _data, float* dst, const float* src),
		void (*der)(void* _data, float* dst, const float* src),
		void (*adj)(void* _data, float* dst, const float* src),
		void* data2,
		long N, float* dst, long M, const float* src);



struct iter3_irgnm_conf {

	INTERFACE(iter3_conf);

	int iter;
	float alpha;
	float redu;
};

extern DEF_TYPEID(iter3_irgnm_conf);

iter3_fun_f iter3_irgnm;



struct iter3_landweber_conf {

	INTERFACE(iter3_conf);

	int iter;
	float alpha;
	float epsilon;
};

extern DEF_TYPEID(iter3_landweber_conf);

iter3_fun_f iter3_landweber;



