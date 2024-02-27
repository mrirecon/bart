/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Philip Schaten <philip.schaten@tugraz.at>
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/misc.h"

#include "lock.h"

struct bart_lock {
#ifdef _OPENMP
	omp_lock_t omp;
#else
	int dummy;
#endif
};

void bart_lock(bart_lock_t* lock)
{
#ifdef _OPENMP
	omp_set_lock(&lock->omp);
#endif
}

void bart_unlock(bart_lock_t* lock)
{
#ifdef _OPENMP
	omp_unset_lock(&lock->omp);
#endif
}

bart_lock_t* bart_lock_create(void)
{
	bart_lock_t* lock = xmalloc(sizeof *lock);

#ifdef _OPENMP
	omp_init_lock(&lock->omp);
#endif
	return lock;
}

void bart_lock_destroy(bart_lock_t* lock)
{
#ifdef _OPENMP
	omp_destroy_lock(&lock->omp);
#endif
	xfree(lock);
}

