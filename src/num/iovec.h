/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdbool.h>


struct iovec_s {
	
	unsigned int N;
	const long* dims;
	const long* strs;
};


extern const struct iovec_s* iovec_create(unsigned int N, const long dims[N]);
extern const struct iovec_s* iovec_create2(unsigned int N, const long dims[N], const long strs[N]);
extern void iovec_free(const struct iovec_s* x);
extern bool iovec_check(const struct iovec_s* iov, unsigned int N, const long dims[N], const long strs[N]);

extern void debug_print_iovec(int level, const struct iovec_s* vec);



