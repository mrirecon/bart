/* Copyright 2014. The Regents of the University of California.
 * Copyright 2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __NIHT_H
#define __NIHT_H

/**
 * Store information about NIHT algorithm configuration.
 *
 * @param maxiter maximum iteration
 * @param epsilon stopping criterion 
 * @param num_funs number of helper functions
 * @param help_ops helper function operators
 */

struct niht_conf_s {

	unsigned int maxiter;
	float epsilon;
	unsigned int num_funs;
	struct operator_s* help_ops;
};

void niht(const struct niht_conf_s* conf,
	  long N, const struct vec_iter_s* vops,
	  struct iter_op_s op, struct iter_op_p_s thresh,
	  float* x, const float* b,
	  struct iter_monitor_s* monitor);


#endif // __NIHT_H
