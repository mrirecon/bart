/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

extern void num_init(void);
extern void num_init_gpu(void);
/**
 * selects the GPU with the maximum available memory
 *   (if there are more than one on the system)
 */
extern void num_init_gpu_memopt(void);
extern void num_init_gpu_device(int device);
extern void num_set_num_threads(int n);

