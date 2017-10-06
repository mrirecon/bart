/* Copyright 2014. The Regents of the University of California.
 * Copyright 2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __NIHT_H
#define __NIHT_H

/**
 * struct containing linear transform operator for NIHT (e.g. wavelets)
 *
 * @param forward the forward transform operator
 * @param adjoint the adjoint transform operator 
 * @param N length of transform vector in floats
 */
struct niht_transop {

	struct iter_op_s forward;
	struct iter_op_s adjoint;
	long N;
};

/**
 * Store information about NIHT algorithm configuration.
 *
 * @param maxiter maximum iteration
 * @param epsilon stopping criterion 
 * @param N length of image vector in floats
 * @param trans flag for use of transform (0/1)
 * @param do_warmstart flag for initial x vector
 */

struct niht_conf_s {

	unsigned int maxiter;
	float epsilon;
	long N;
	int trans;
	_Bool do_warmstart;
};

void niht(const struct niht_conf_s* conf, const struct niht_transop* trans, 
	  const struct vec_iter_s* vops,
	  struct iter_op_s op, struct iter_op_p_s thresh,
	  float* x, const float* b,
	  struct iter_monitor_s* monitor);


#endif // __NIHT_H
