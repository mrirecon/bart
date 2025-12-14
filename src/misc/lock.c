/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Philip Schaten <philip.schaten@tugraz.at>
 */

#ifdef _WIN32

#include <assert.h>
#include "lock.h"

void bart_lock(bart_lock_t* lock) { assert(0); }
void bart_unlock(bart_lock_t* lock) { assert(0); }
void bart_lock_destroy(bart_lock_t* x) { assert(0); }
bart_lock_t* bart_lock_create(void) { assert(0); }
void bart_cond_wait(bart_cond_t* cond, bart_lock_t* lock) { assert(0); }
void bart_cond_notify_all(bart_cond_t* cond) { assert(0); }
void bart_cond_destroy(bart_cond_t* x) { assert(0); }
bart_cond_t* bart_cond_create(void) { assert(0); }

#else

#ifdef __APPLE__
#	include <pthread.h>
#else
#	include <threads.h>
#endif

#include <stdbool.h>

#include "misc/misc.h"

#include "lock.h"

struct bart_lock {
#ifdef __APPLE__
	pthread_mutex_t mx;
#else
	mtx_t mx;
#endif
};

void bart_lock(bart_lock_t* lock)
{
#ifdef __APPLE__
	pthread_mutex_lock(&lock->mx);
#else
	mtx_lock(&lock->mx);
#endif
}

//returns true if lock was acquired
bool bart_trylock(bart_lock_t* lock)
{
#ifdef __APPLE__
	return 0 == pthread_mutex_trylock(&lock->mx);
#else
	return 0 == mtx_trylock(&lock->mx);
#endif
}

void bart_unlock(bart_lock_t* lock)
{
#ifdef __APPLE__
	pthread_mutex_unlock(&lock->mx);
#else
	mtx_unlock(&lock->mx);
#endif
}

bart_lock_t* bart_lock_create(void)
{
	bart_lock_t* lock = xmalloc(sizeof *lock);
#ifdef __APPLE__
	pthread_mutex_init(&lock->mx, PTHREAD_MUTEX_DEFAULT);
#else
	mtx_init(&lock->mx, mtx_plain);
#endif

	return lock;
}

void bart_lock_destroy(bart_lock_t* lock)
{
#ifdef __APPLE__
	pthread_mutex_destroy(&lock->mx);
#else
	mtx_destroy(&lock->mx);
#endif

	xfree(lock);
}


struct bart_cond {

#ifdef __APPLE__
	long counter;
	pthread_cond_t cnd;
#else
	long counter;
	cnd_t cnd;
#endif
};

bart_cond_t* bart_cond_create(void)
{
	bart_cond_t* cond = xmalloc(sizeof *cond);

#ifdef __APPLE__
	pthread_cond_init(&cond->cnd, NULL);
	cond->counter = 0;
#else
	cnd_init(&cond->cnd);
	cond->counter = 0;
#endif

	return cond;
}

void bart_cond_wait(bart_cond_t* cond, bart_lock_t* lock)
{
	long counter = cond->counter;

#ifdef __APPLE__
	while (counter == cond->counter)
		pthread_cond_wait(&cond->cnd, &lock->mx);
#else
	while (counter == cond->counter)
		cnd_wait(&cond->cnd, &lock->mx);
#endif
}

void bart_cond_notify_all(bart_cond_t* cond)
{
	cond->counter++;
#ifdef __APPLE__
	pthread_cond_broadcast(&cond->cnd);
#else
	cnd_broadcast(&cond->cnd);
#endif
}

void bart_cond_destroy(bart_cond_t* cond)
{
#ifdef __APPLE__
	pthread_cond_destroy(&cond->cnd);
#else
	cnd_destroy(&cond->cnd);
#endif
	xfree(cond);
}

#endif
