void blas_zfmac_cgemm(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cgemv(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_caxpy(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cgeru(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zfmac_cdotu(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

void blas_fmac_sgemm(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sgemv(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_fmac_saxpy(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sger(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_fmac_sdot(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);

void blas_zmul_cmatcopy(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zsmul_cmatcopy(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr[__VLA(N)], const _Complex float* iptr, _Complex float val);
void blas_zmul_cdgmm(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zmul_cgeru(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void blas_zmul_cscal(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

void blas_mul_smatcopy(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_smul_smatcopy(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr[__VLA(N)], const float* iptr, float val);
void blas_mul_sdgmm(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_mul_sger(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void blas_mul_sscal(unsigned int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
