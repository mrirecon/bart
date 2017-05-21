/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __SENSE_H
#define __SENSE_H 1

#include "misc/mri.h"
#include "iter/iter.h"
#include "iter/iter2.h"

#include "misc/cppwrap.h"




/**
 * configuration parameters for sense reconstruction
 *
 * @param rvc TRUE for real-values constraints
 */
struct sense_conf {

	_Bool rvc;
	_Bool gpu;
	int rwiter;	// should be moved into a recon_lad
	float gamma;	// ..
	float cclambda;
};


extern const struct sense_conf sense_defaults;

struct operator_s;
struct operator_p_s;

extern const struct operator_s* sense_recon_create(const struct sense_conf* conf, const long dims[DIMS],
		  const struct linop_s* sense_op,
		  const long pat_dims[DIMS], const complex float* pattern,
		  italgo_fun2_t italgo, iter_conf* iconf,
		  const complex float* init,
		  unsigned int num_funs,
		  const struct operator_p_s* thresh_op[num_funs],
		  const struct linop_s* thresh_funs[num_funs],
		  const struct operator_s* precond_op);

extern void debug_print_sense_conf(int debug_level, const struct sense_conf* conf);



#include "misc/cppwrap.h"

#endif


