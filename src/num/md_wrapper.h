void zfmac_gpu_batched_loop(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void zfmacc_gpu_batched_loop(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

void add_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void zadd_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

void mul_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void zmul_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void zmulc_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

void fmac_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
void zfmac_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
void zfmacc_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
