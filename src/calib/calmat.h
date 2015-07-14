
#include "misc/cppwrap.h"

extern complex float* calibration_matrix(long calmat_dims[2], const long kdims[3], const long calreg_dims[4], const complex float* data);
extern complex float* calibration_matrix_mask(long calmat_dims[2], const long kdims[3], const complex float* mask, const long calreg_dims[4], const complex float* data);
extern complex float* calibration_matrix_mask2(long calmat_dims[2], const long kdims[3], const complex float* msk, const long calreg_dims[4], const complex float* data);

#ifndef __cplusplus
extern void covariance_function(const long kdims[3], unsigned int N, complex float cov[static N][N], const long calreg_dims[4], const complex float* data);
extern void calmat_svd(const long kdims[3], unsigned int N, complex float cov[static N][N], float* S, const long calreg_dims[4], const complex float* data);
#endif

#include "misc/cppwrap.h"

