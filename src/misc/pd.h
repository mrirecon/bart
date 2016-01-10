/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef __cplusplus
#error This file does not support C++
#endif

extern int poissondisc(int D, int N, int II, float vardens, float delta, float points[static N][D]);
extern int poissondisc_mc(int D, int T, int N, int II, float vardens,
	const float delta[static T][T], float points[static N][D], int kind[static N]);

extern void mc_poisson_rmatrix(int D, int T, float rmatrix[static T][T], const float delta[static T]);

#if __GNUC__ < 5
#include "misc/pcaa.h"

#define poissondisc_mc(A, B, C, D, E, x, y, z) \
	poissondisc_mc(A, B, C, D, E, AR2D_CAST(float, B, B, x), y, z)

#endif

