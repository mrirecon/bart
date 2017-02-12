/* Copyright 2014-2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __OPTREG_H
#define __OPTREG_H

#include "misc/cppwrap.h"


#define NUM_REGS 10

struct operator_p_s;
struct linop_s;

enum algo_t { CG, IST, FISTA, ADMM };

struct reg_s {

	enum { L1WAV, TV, LLR, MLR, IMAGL1, IMAGL2, L1IMG, L2IMG, FTL1, LAPLACE } xform;

	unsigned int xflags;
	unsigned int jflags;

	float lambda;
};


struct opt_reg_s {

	float lambda;
	enum algo_t algo;
	struct reg_s regs[NUM_REGS];
	unsigned int r;
};



extern _Bool opt_reg_init(struct opt_reg_s* ropts);

extern void opt_reg_configure(unsigned int N, const long img_dims[__VLA(N)], struct opt_reg_s* ropts, const struct operator_p_s* thresh_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int llr_blk, _Bool randshift, _Bool use_gpu);

extern _Bool opt_reg(void* ptr, char c, const char* optarg);

extern void help_reg(void);

#include "misc/cppwrap.h"
#endif
