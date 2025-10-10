#include "misc/cppwrap.h"

extern float gaussian_pdf(int N, const complex float m[N],
		const complex float icov[N][N], const complex float x[N]);
extern void gaussian_sample(int N, const complex float m[N],
		const complex float icov[N][N], complex float x[N]);
extern float gaussian_mix_pdf(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float icov[M][N][N], const complex float x[N]);
extern void gaussian_mix_sample(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float sqr_cov[M][N][N], complex float x[N]);
extern void gaussian_score(int N, const complex float m[N], const complex float icov[N][N],
		const complex float x[N], complex float sc[N]);
extern void gaussian_mix_score(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float icov[M][N][N], const complex float x[N], complex float sc[N]);
extern void gaussian_multiply(int N, complex float m[N], complex float icov[N][N],
		const complex float m1[N], const complex float icov1[N][N],
		const complex float m2[N], const complex float icov2[N][N]);
extern float gaussian_multiply_factor(int N,
		const complex float m1[N], const complex float icov1[N][N],
		const complex float m2[N], const complex float icov2[N][N]);
extern void gaussian_convolve(int N, complex float m[N], complex float sqr_cov[N][N],
		const complex float m1[N], const complex float sqr_cov1[N][N],
		const complex float m2[N], const complex float sqr_cov2[N][N]);

extern void md_grad_gaussian(int D, const long dims_grad[__VLA(D)], complex float* grad,
		const long dims_x[__VLA(D)], const complex float* x, const long dims_mu[__VLA(D)],
		const complex float* mu, const long dims_sigma[__VLA(D)], const complex float* sigma);

extern void md_log_gaussian(int D, const long dims_log_gauss[__VLA(D)], complex float* log_gauss,
		const long dims_x[__VLA(D)], const complex float* x,
		const long dims_mu[__VLA(D)], const complex float* mu,
		const long dims_sigma[__VLA(D)], const complex float* sigma);

extern void md_mixture_weights(int D, const long dims_gamma[__VLA(D)], complex float* gamma,
		const long dims_log_gauss[__VLA(D)], complex float* log_gauss,
		const long dims_ws[__VLA(D)], const complex float* ws);

extern void md_gaussian_score(int D, const long dims_score[__VLA(D)], complex float* score,
		const long dims_x[__VLA(D)], const complex float* x,
		const long dims_mu[__VLA(D)], const complex float* mu,
		const long dims_sigma[__VLA(D)], const complex float* sigma,
		const long dims_ws[__VLA(D)], const complex float* ws);

