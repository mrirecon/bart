/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */


struct fft_cuda_plan_s;

extern struct fft_cuda_plan_s* fft_cuda_plan(unsigned int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], _Bool dir);
extern void fft_cuda_free_plan(struct fft_cuda_plan_s* cuplan);
extern void fft_cuda_exec(struct fft_cuda_plan_s* cuplan, complex float* dst, const complex float* src);

