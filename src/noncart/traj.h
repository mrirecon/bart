/* Copyright 2014-2015 The Regents of the University of California.
 * Copyright 2015-2019 Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2018-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

struct traj_conf {

	_Bool radial;
	_Bool golden;
	_Bool aligned;
	_Bool full_circle;
	_Bool half_circle_gold;
	_Bool golden_partition;
	_Bool d3d;
	_Bool transverse;
	_Bool asym_traj;
	_Bool mems_traj;
	_Bool rational;
	_Bool double_base;
	int accel;
	int tiny_gold;
	int turns;
	int mb;
};

extern const struct traj_conf traj_defaults;
extern const struct traj_conf rmfreq_defaults;

#ifndef DIMS
#define DIMS 16
#endif

extern void euler(float dir[3], float phi, float psi);
extern void gradient_delay(float d[3], float coeff[2][3], float phi, float psi);
extern void calc_base_angles(double base_angle[DIMS], int Y, int E, struct traj_conf conf);
extern void indices_from_position(long ind[DIMS], const long pos[DIMS], struct traj_conf conf, long start_pos_GA);
extern bool zpartition_skip(long partitions, long z_usamp[2], long partition, long frame);
extern int gen_fibonacci(int n, int ind);
extern int recover_gen_fib_ind(int Y, int inc);
extern int raga_increment(int Y, int n);

