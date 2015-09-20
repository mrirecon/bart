/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
extern "C" {
#define __VLA(x) 
#else
#define __VLA(x) static x
#endif



extern int write_coo(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_coo(int fd, int n, long dimensions[__VLA(n)]);

extern int write_cfl_header(int fd, int n, const long dimensions[__VLA(n)]);
extern int read_cfl_header(const char* tag, int fd, int D, long dimensions[__VLA(D)]);



#ifdef __cplusplus
}
#endif

