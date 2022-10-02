void blas_zfmac_cgemm(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cgemv(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_caxpy(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cgeru(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cdotu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

void blas_fmac_sgemm(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sgemv(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_fmac_saxpy(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sger(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sdot(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);

void blas_zmul_cmatcopy(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zsmul_cmatcopy(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr[__VLA(N)], const _Complex float* iptr, _Complex float val);
void blas_zmul_cdgmm(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zmul_cgeru(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zmul_cscal(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

void blas_mul_smatcopy(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_smul_smatcopy(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr[__VLA(N)], const float* iptr, float val);
void blas_mul_sdgmm(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_mul_sger(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_mul_sscal(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
