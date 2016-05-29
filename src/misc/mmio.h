/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/cppwrap.h"

// extern void* private_raw(size_t* size, const char* name);

extern _Complex float* shared_cfl(unsigned int D, const long dims[__VLA(D)], const char* name);
extern _Complex float* private_cfl(unsigned int D, const long dims[__VLA(D)], const char* name);
extern void unmap_cfl(unsigned int D, const long dims[__VLA(D)], const _Complex float* x);

extern _Complex float* anon_cfl(const char* name, unsigned int D, const long dims[__VLA(D)]);
extern _Complex float* create_cfl(const char* name, unsigned int D, const long dimensions[__VLA(D)]);
extern _Complex float* load_cfl(const char* name, unsigned int D, long dimensions[__VLA(D)]);
extern _Complex float* load_shared_cfl(const char* name, unsigned int D, long dimensions[__VLA(D)]);

extern float* create_coo(const char* name, unsigned int D, const long dimensions[__VLA(D)]);
extern float* load_coo(const char* name, unsigned int D, long dimensions[__VLA(D)]);
extern _Complex float* create_zcoo(const char* name, unsigned int D, const long dimensions[__VLA(D)]);
extern _Complex float* load_zcoo(const char* name, unsigned int D, long dimensions[__VLA(D)]);
extern _Complex float* create_zra(const char* name, unsigned int D, const long dims[__VLA(D)]);
extern _Complex float* load_zra(const char* name, unsigned int D, long dims[__VLA(D)]);

#include "misc/cppwrap.h"


