/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Philip Schaten <philip.schaten@tugraz.at>
 */

#ifdef _OPENMP
#	ifdef __APPLE__
#		include <pthread.h>
#	else
#		include <threads.h>
#	endif
#endif

#include "misc/misc.h"

#include "lock.h"

struct bart_lock {
#ifdef _OPENMP
#ifdef __APPLE__
	pthread_mutex_t mx;
#else
	mtx_t mx;
#endif
#else
	int dummy;
#endif
};

void bart_lock(bart_lock_t* lock)
{
#ifdef _OPENMP
#ifdef __APPLE__
	pthread_mutex_lock(&lock->mx);
#else
	mtx_lock(&lock->mx);
#endif

#else
	(void)lock;
#endif
}

void bart_unlock(bart_lock_t* lock)
{
#ifdef _OPENMP
#ifdef __APPLE__
	pthread_mutex_unlock(&lock->mx);
#else
	mtx_unlock(&lock->mx);
#endif

#else
	(void)lock;
#endif
}

bart_lock_t* bart_lock_create(void)
{
	bart_lock_t* lock = xmalloc(sizeof *lock);

#ifdef _OPENMP
#ifdef __APPLE__
	pthread_mutex_init(&lock->mx, PTHREAD_MUTEX_DEFAULT);
#else
	mtx_init(&lock->mx, mtx_plain);
#endif

#endif
	return lock;
}

void bart_lock_destroy(bart_lock_t* lock)
{
#ifdef _OPENMP
#ifdef __APPLE__
	pthread_mutex_destroy(&lock->mx);
#else
	mtx_destroy(&lock->mx);
#endif

#endif
	xfree(lock);
}


struct bart_cond {

#ifdef _OPENMP
#ifdef __APPLE__
	long counter;
	pthread_cond_t cnd;
#else
	long counter;
	cnd_t cnd;
#endif

#else
	int dummy;
#endif
};

bart_cond_t* bart_cond_create(void)
{
	bart_cond_t* cond = xmalloc(sizeof *cond);

#ifdef _OPENMP
#ifdef __APPLE__
	pthread_cond_init(&cond->cnd, NULL);
	cond->counter = 0;
#else
	cnd_init(&cond->cnd);
	cond->counter = 0;
#endif

#endif
	return cond;
}

void bart_cond_wait(bart_cond_t* cond, bart_lock_t* lock)
{
#ifdef _OPENMP
	long counter = cond->counter;

#ifdef __APPLE__
	while (counter == cond->counter)
		pthread_cond_wait(&cond->cnd, &lock->mx);
#else
	while (counter == cond->counter)
		cnd_wait(&cond->cnd, &lock->mx);
#endif

#else
	(void)cond;
	(void)lock;
#endif
}

void bart_cond_notify_all(bart_cond_t* cond)
{
#ifdef _OPENMP
	cond->counter++;
#ifdef __APPLE__
	pthread_cond_broadcast(&cond->cnd);
#else
	cnd_broadcast(&cond->cnd);
#endif
#else
	(void)cond;
#endif
}

void bart_cond_destroy(bart_cond_t* cond)
{
#ifdef _OPENMP
#ifdef __APPLE__
	pthread_cond_destroy(&cond->cnd);
#else
	cnd_destroy(&cond->cnd);
#endif

#endif
	xfree(cond);
}

