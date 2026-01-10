/* Copyright 2015. The Regents of the University of California.
 * Copyright 2017,2022. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2018. Massachusetts Institute of Technology.
 * Copyright 2024. Institute of Biomedical Imaging. TU Graz.
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
#include "misc/debug.h"

#include "iter/italgos.h"
#include "iter/iter2.h"
#include "iter/vec.h"

#include "misc.h"





double iter_power(int maxiter,
		const struct operator_s* normaleq_op,
		long size, float* u)
{
	return power(maxiter, size, select_vecops(u), OPERATOR2ITOP(normaleq_op), u, NULL);
}

double estimate_maxeigenval_sameplace(const struct operator_s* op, int iterations, const void *ref)
{
	const struct iovec_s* io = operator_domain(op);
	long size = md_calc_size(io->N, io->dims);

	if (NULL == ref)
		ref = &size; // cpu_ref

	void* x = md_alloc_sameplace(io->N, io->dims, io->size, ref);
	void* b = md_alloc_sameplace(io->N, io->dims, io->size, ref);

	select_vecops(ref)->rand(2 * size, x);

	double max_eval = power(iterations, 2 * size, select_vecops(x), OPERATOR2ITOP(op), x, b);

	debug_printf(DP_DEBUG1, "Maximum eigenvalue: %e\n", max_eval);

	md_free(x);
	md_free(b);

	return max_eval;
}


double estimate_maxeigenval(const struct operator_s* op)
{
	return estimate_maxeigenval_sameplace(op, 30, NULL);
}

#ifdef USE_CUDA
double estimate_maxeigenval_gpu(const struct operator_s* op)
{
	void* ref = md_alloc_gpu(1, MD_DIMS(1), 1);

	double max_eval = estimate_maxeigenval_sameplace(op, 30, ref);

	md_free(ref);

	return max_eval;
}
#endif
