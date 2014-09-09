/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 

typedef void iter3_fun_f(void* _conf,
		void (*frw)(void* _data, float* dst, const float* src),
		void (*der)(void* _data, float* dst, const float* src),
		void (*adj)(void* _data, float* dst, const float* src),
		void* data2,
		long N, float* dst, long M, const float* src);



struct iter3_irgnm_conf {

	int iter;
	float alpha;
	float redu;
};

iter3_fun_f iter3_irgnm;



struct iter3_landweber_conf {

	int iter;
	float alpha;
	float epsilon;
};

iter3_fun_f iter3_landweber;



