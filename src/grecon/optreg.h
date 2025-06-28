/* Copyright 2014-2017. The Regents of the University of California.
 * Copyright 2018-2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _OPTREG_H
#define _OPTREG_H

#include "misc/cppwrap.h"


#define NUM_REGS 10
#define NUM_TV_SCALES 5

struct operator_p_s;
struct linop_s;


struct reg_s {

	enum { L1WAV, NIHTWAV, NIHTIM, TV, LLR, MLR, IMAGL1, IMAGL2, L1IMG, L2IMG, FTL1, LAPLACE, POS, TENFL, TGV, ICTV, ICTGV } xform;

	unsigned long xflags;
	unsigned long jflags;

	float lambda;
	int k;
	const char* graph_file;

	_Bool asl;
};


struct opt_reg_s {

	float lambda;
	struct reg_s regs[NUM_REGS];
	int r;
	long svars;
	int sr;
	
	int tvscales_N;
	float tvscales[NUM_TV_SCALES];

	int tvscales2_N;
	float tvscales2[NUM_TV_SCALES];

	_Bool asl;
	_Bool teasl;

	float theta[2];

	float alpha[2];
	float gamma[2];
};



extern _Bool opt_reg_init(struct opt_reg_s* ropts);

extern void opt_bpursuit_configure(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], const struct linop_s* model_op, const _Complex float* data, const float eps);
extern void opt_precond_configure(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], const struct linop_s* model_op, int N, const long ksp_dims[N], const _Complex float* data, const long pat_dims[N], const _Complex float* pattern);

extern void opt_reg_configure(int N, const long img_dims[__VLA(N)], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], const long (*sdims[NUM_REGS])[N + 1], int llr_blk, int shift_mode, const char* wtype_str, _Bool use_gpu, int asl_dim);

extern void opt_reg_free(struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS]);

extern _Bool opt_reg(void* ptr, char c, const char* optarg);

extern void help_reg(void);

#include "misc/cppwrap.h"
#endif
