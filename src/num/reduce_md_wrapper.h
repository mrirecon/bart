extern void reduce_zadd_inner_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
extern void reduce_zadd_outer_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
extern void reduce_zadd_gemv(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

extern void reduce_add_inner_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
extern void reduce_add_outer_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
extern void reduce_add_gemv(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);

extern void reduce_zmax_inner_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
extern void reduce_zmax_outer_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);

