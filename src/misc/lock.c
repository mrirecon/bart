/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Philip Schaten <philip.schaten@tugraz.at>
 */

#ifdef _OPENMP
#include <threads.h>
#endif

#include "misc/misc.h"

#include "lock.h"

struct bart_lock {
#ifdef _OPENMP
	mtx_t mx;
#else
	int dummy;
#endif
};

void bart_lock(bart_lock_t* lock)
{
#ifdef _OPENMP
	mtx_lock(&lock->mx);
#else
	(void)lock;
#endif
}

void bart_unlock(bart_lock_t* lock)
{
#ifdef _OPENMP
	mtx_unlock(&lock->mx);
#else
	(void)lock;
#endif
}

bart_lock_t* bart_lock_create(void)
{
	bart_lock_t* lock = xmalloc(sizeof *lock);

#ifdef _OPENMP
	mtx_init(&lock->mx, mtx_plain);
#endif
	return lock;
}

void bart_lock_destroy(bart_lock_t* lock)
{
#ifdef _OPENMP
	mtx_destroy(&lock->mx);
#endif
	xfree(lock);
}


struct bart_cond {

#ifdef _OPENMP
	long counter;
	cnd_t cnd;
#else
	int dummy;
#endif
};

bart_cond_t* bart_cond_create(void)
{
	bart_cond_t* cond = xmalloc(sizeof *cond);

#ifdef _OPENMP
	cnd_init(&cond->cnd);
	cond->counter = 0;
#endif
	return cond;
}

void bart_cond_wait(bart_cond_t* cond, bart_lock_t* lock)
{
#ifdef _OPENMP
	long counter = cond->counter;

	while (counter == cond->counter)
		cnd_wait(&cond->cnd, &lock->mx);
#else
	(void)cond;
	(void)lock;
#endif
}

void bart_cond_notify_all(bart_cond_t* cond)
{
#ifdef _OPENMP
	cond->counter++;
	cnd_broadcast(&cond->cnd);
#endif
}

void bart_cond_destroy(bart_cond_t* cond)
{
#ifdef _OPENMP
	cnd_destroy(&cond->cnd);
#endif
	xfree(cond);
}

