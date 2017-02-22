/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 */
 
#include "num/multind.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/rand.h"

#include "misc/misc.h"

#include "iter/italgos.h"
#include "iter/iter2.h"
#include "iter/vec.h"

#include "misc.h"





double iter_power(unsigned int maxiter,
		const struct operator_s* normaleq_op,
		long size, float* u)
{
	return power(maxiter, size, select_vecops(u), OPERATOR2ITOP(normaleq_op), u);
}


double estimate_maxeigenval(const struct operator_s* op)
{
	const struct iovec_s* io = operator_domain(op);
	long size = md_calc_size(io->N, io->dims);

	void* x = md_alloc(io->N, io->dims, io->size);

	md_gaussian_rand(io->N, io->dims, x);

	double max_eval = iter_power(30, op, 2 * size, x);

	md_free(x);

	return max_eval;
}

