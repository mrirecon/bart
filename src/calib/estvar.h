/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Siddharth Iyer <sid8795@gmail.com>
 */

#ifndef __ESTVAR_H
#define __ESTVAR_H
 
#define TOOLBOX_PATH getenv("TOOLBOX_PATH")

/**
 * estvar_sv - This estimates the variance of noise present in the 
 *             calibration region using the singular values of the
 *             calibration matrix.
 * 
 * Parameters:
 *  L           - Number of singular values.
 *  S           - Array of singular values.
 *  kernel_dims - Kernel dimensions.
 *  calreg_dims - Calibration region dimensions.
 */
extern float estvar_sv(long L, const float S[L], const long kernel_dims[3], const long calreg_dims[4]);

/**
 * estvar_calreg - This estimates the variance of noise present in the 
 *                calibration region.
 * 
 * Parameters:
 *  kernel_dims - Kernel dimensions.
 *  calreg_dims - Calibration region dimensions.
 *  calreg      - Calibration region.
 */
extern float estvar_calreg(const long kernel_dims[3], const long calreg_dims[4], const complex float* calreg);

/**
 * estvar_kspace - This estimates the variance of noise present in kspace data.
 * 
 * Parameters:
 *  N           - Total number of dimensions in a CFL file.
 *  kernel_dims - Kernel dimensions.
 *  calib_size  - Size of the calibration region.
 *  kspace_dims - Dimensions of input data.
 *  kspace      - Input kspace data.
 */
extern float estvar_kspace(long N, const long kernel_dims[3], const long calib_size[3], const long kspace_dims[N], const complex float* kspace);

#endif
