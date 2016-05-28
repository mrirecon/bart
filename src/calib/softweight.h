/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015-2016 Siddharth Iyer <sid8795@gmail.com>
 *
 * Iyer S, Ong F, Lustig M.
 * Towards A Parameter­Free ESPIRiT: Soft­Weighting For Robust Coil Sensitivity Estimation.
 * In Proceedings Of ISMRM 2016.
 *
 * Candès E, Long C, Trzasko J. 
 * Unbiased Risk Estimates for Singular Value Thresholding and Spectral Estimators.
 * IEEE Transactions on Signal Processing 61, no. 19 (2013): 4643­657.
 *
 */

#ifndef _SOFT_WEIGHT_H_
#define _SOFT_WEIGHT_H_
 
/**
 * soft_weight_singular_vectors - This returns weights for the singular vectors derived from the 
 *                                soft-thresholding operator proposed by Candès et al., as seen
 *                                in "Towards a Parameter­Free ESPIRiT: Soft­Weighting for 
 *                                Robust Coil Sensitivity Estimation."
 * 
 * Parameters:
 *  N           - Number of singular values.
 *  kernel_dims - Dimension of kernel.
 *  calreg_dims - Calibration region dimensions.
 *  S           - Array of singular values. 
 *  W           - Array to store weights to.
 */
extern void soft_weight_singular_vectors(long N, float var, const long kernel_dims[3], const long calreg_dims[4], const float S[N], float W[N]);

#endif
