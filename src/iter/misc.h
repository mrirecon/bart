/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

struct operator_s;
extern double iter_power(unsigned int maxiter,
		const struct operator_s* normaleq_op,
		long size, float* u);

extern double estimate_maxeigenval(const struct operator_s* op);


