
extern float* chebfun(int* NP, float (*fun)(float x));
extern void chebfun2(int N, float coeff[N], float (*fun)(float x));
extern void chebpoly(int N, float coeff[N], const float val[N]);
extern void chebinv(int N, float val[N], const float coeff[N]);
extern void chebadd(int A, int B, float dst[(A > B) ? A : B], const float src1[A], const float src2[B]);
extern void chebmul(int A, int B, float dst[A + B], const float src1[A], const float src2[B]);
extern float chebint(int N, const float coeff[N]);
extern float chebeval(float x, int N, const float pval[N]);
extern void chebdiff(int N, float dst[N - 1], const float src[N]);
extern void chebindint(int N, float dst[N + 1], const float src[N]);


