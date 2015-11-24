/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Siddharth Iyer <sid8795@gmail.com>
 *
 * Iyer S, Ong F, Lustig M.
 * Towards a Parameter­Free ESPIRiT: Soft­Weighting for Robust Coil Sensitivity Estimation
 * Submitted to ISMRM 2016.
 */

#ifndef _SOFT_WEIGHT_H_
#define  _SOFT_WEIGHT_H_
 
//TODO: Document
extern void soft_weight_singular_vectors(long N, long kernel_dims[3], long calreg_dims[4], float S[N]);

#endif
