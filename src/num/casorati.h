/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

extern void casorati_dims(unsigned int N, long odim[2], const long dimk[N], const long dims[N]);
extern void casorati_matrix(unsigned int N, const long dimk[N], const long odim[2], _Complex float* optr, const long dim[N], const long str[N], const _Complex float* iptr);
extern void casorati_matrixH(unsigned int N, const long dimk[N], const long dim[N], const long str[N], _Complex float* optr, const long odim[2], const _Complex float* iptr);

