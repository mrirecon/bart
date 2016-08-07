/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2016 Siddharth Iyer <sid8795@gmail.com>
 */

#ifndef __ESTCOV_H
#define __ESTCOV_H
 
/**
 * estcov - This numerically estimates the noise covariance matrix between coils
 *          using the edges of kspace data which is assumed to dominantly contain 
 *          noise.
 *
 * Parameters:
 *  out_dims    - [nc x nc] Dimensions of the output covariance matrix. nc refers 
 *                to the number of coils.
 *  out_data    - Output covariance matrix.
 *  p           - Parameter that determines the edge thickness. For example, if 
 *                p = 0.1 and the length of the kx direction is 100, then a width
 *                of p * kx = 10 is used.
 *  kspace_dims - [kx x ky x kz x nc] Dimensions of the input kspace data.
 *  kspace      - Input kspace data.
 */
extern void estcov(long out_dims[2], complex float* out_data, float p, long N, const long kspace_dims[N], const complex float* kspace);

/**
 * estcov_var - This used estcov to estimate the variance of the data.
 *
 * Parameters:
 *  p           - Parameter that determines the edge thickness. For example, if 
 *                p = 0.1 and the length of the kx direction is 100, then a width
 *                of p * kx = 10 is used.
 *  kspace_dims - [kx x ky x kz x nc] Dimensions of the input kspace data.
 *  kspace      - Input kspace data.
 */
extern float estcov_var(float p, long N, const long kspace_dims[N], const complex float* kspace);

#endif
