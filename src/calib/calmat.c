
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/casorati.h"
#include "num/la.h"

#include "misc/mri.h"

#include "calmat.h"


static complex float* calibration_matrix_priv(long calmat_dims[2], const long kdims[3], const long calreg_dims[4], const complex float* data)
{
	long kernel_dims[4];
	md_copy_dims(3, kernel_dims, kdims);
	kernel_dims[3] = calreg_dims[3];

	casorati_dims(4, calmat_dims, kernel_dims, calreg_dims);

	complex float* cm = md_alloc_sameplace(2, calmat_dims, CFL_SIZE, data);

	long calreg_strs[4];
	md_calc_strides(4, calreg_strs, calreg_dims, CFL_SIZE);
	casorati_matrix(4, kernel_dims, calmat_dims, cm, calreg_dims, calreg_strs, data);

	return cm;
}


complex float* calibration_matrix(long calmat_dims[2], const long kdims[3], const long calreg_dims[4], const complex float* data)
{
#if 1
	return calibration_matrix_priv(calmat_dims, kdims, calreg_dims, data);
#else
	// estimate pattern
	long pat_dims[4];
	md_select_dims(4, ~(1 << COIL_DIM), pat_dims, calreg_dims);
	complex float* pattern = md_alloc_sameplace(4, pat_dims, CFL_SIZE, data);
	estimate_pattern(4, calreg_dims, COIL_DIM, pattern, data);

	// compute calibration matrix of pattern
	long pcm_dims[2];
	complex float* pm = calibration_matrix_priv(pcm_dims, kdims, pat_dims, pattern);
	md_free(pattern);

	// number of samples for each patch
	long pcm_strs[2];
	md_calc_strides(2, pcm_strs, pcm_dims, CFL_SIZE);

	long msk_dims[2];
	md_select_dims(2, ~(1 << 1), msk_dims, pcm_dims);

	long msk_strs[2];
	md_calc_strides(2, msk_strs, msk_dims, CFL_SIZE);

	complex float* msk = md_alloc(2, msk_dims, CFL_SIZE);
	md_clear(2, msk_dims, msk, CFL_SIZE);
	md_zfmacc2(2, pcm_dims, msk_strs, msk, pcm_strs, pm, pcm_strs, pm);
	md_free(pm);

	// fully sampled?
	long strs1[2] = { 0, 0 };
	md_zcmp2(2, msk_dims, msk_strs, msk, msk_strs, msk, strs1, &(complex float){ pcm_dims[1] });

	complex float* tmp = calibration_matrix_priv(calmat_dims, kdims, calreg_dims, data);

	// mask out incompletely sampled patches...
	long calmat_strs[2];
	md_calc_strides(2, calmat_strs, calmat_dims, CFL_SIZE);
	md_zmul2(2, calmat_dims, calmat_strs, tmp, calmat_strs, tmp, msk_strs, msk);

	return tmp;
#endif
}


extern void covariance_function(const long kdims[3], complex float* cov, const long calreg_dims[4], const complex float* data)
{
	long calmat_dims[2];
	complex float* cm = calibration_matrix(calmat_dims, kdims, calreg_dims, data);

	int L = calmat_dims[0];
	int N = calmat_dims[1];

	gram_matrix(N, (complex float (*)[N])cov, L, (const complex float (*)[L])cm);

	md_free(cm);
}



#ifdef CALMAT_SVD
static void calmat_svd(const long kdims[3], complex float* cov, float* S, const long calreg_dims[4], const complex float* data)
{
	long calmat_dims[2];
	complex float* cm = calibration_matrix(calmat_dims, kdims, calreg_dims, data);

	int L = calmat_dims[0];
	int N = calmat_dims[1];

	complex float* U = xmalloc(L * L * CFL_SIZE);	

	svd(L, N, (complex float (*)[L])U, (complex float (*)[N])cov, S, (complex float (*)[])cm); // why not const last arg

	free(U);
	md_free(cm);
}
#endif
