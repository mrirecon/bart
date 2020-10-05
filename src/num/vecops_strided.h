extern void activate_strided_vecops(void);
extern void deactivate_strided_vecops(void);

extern _Bool simple_zfmac(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], _Complex float* out, const long istrs1[__VLA(N)], const _Complex float* in1, const long istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_zfmacc(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], _Complex float* out, const long istrs1[__VLA(N)], const _Complex float* in1, const long istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_fmac(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], float* out, const long istrs1[__VLA(N)], const float* in1, const long istrs2[__VLA(N)], const float* in2);

extern _Bool simple_zmul(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], _Complex float* out, const long istrs1[__VLA(N)], const _Complex float* in1, const long istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_zmulc(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], _Complex float* out, const long istrs1[__VLA(N)], const _Complex float* in1, const long istrs2[__VLA(N)], const _Complex float* in2);
extern _Bool simple_mul(unsigned int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], float* out, const long istrs1[__VLA(N)], const float* in1, const long istrs2[__VLA(N)], const float* in2);
