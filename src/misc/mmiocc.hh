/* Copyright 2017-2018. Damien Nguyen.
 * Copyright 2017-2018. Francesco Santini
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 * 2017-2018 Francesco Santini <francesco.santini@unibas.ch>
 */

#include "misc/cppwrap.h"

extern void* create_mem_cfl(const char* name, unsigned int D, const long dimensions[]);
extern void* create_anon_mem_cfl(unsigned int D, const long dimensions[]);
extern void* load_mem_cfl(const char* name, unsigned int D, long dimensions[]);

extern void register_mem_cfl_non_managed(const char* name, unsigned int D, const long dims[], void* data);
// Note: for both function below, the internal memory handler takes ownership of the data! (ie. will get automatically deallocated)
extern void register_mem_cfl_malloc(const char* name, unsigned int D, const long dimensions[], void* data);
extern void register_mem_cfl_new(const char* name, unsigned int D, const long dimensions[], void* data);

extern _Bool is_mem_cfl(const _Complex float* ptr);

// Try to delete an entry from the list of memory CFL files.
extern _Bool try_delete_mem_cfl(const _Complex float* ptr);

extern _Bool deallocate_mem_cfl_name(const char* name);
extern _Bool deallocate_mem_cfl_ptr(const _Complex float* ptr);
extern void deallocate_all_mem_cfl();


#include "misc/cppwrap.h"
