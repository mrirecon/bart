
extern float gaussian_pdf(int N, const complex float m[N],
		const complex float isqr_cov[N][N], const complex float x[N]);
extern void gaussian_sample(int N, const complex float m[N],
		const complex float isqr_cov[N][N], complex float x[N]);
extern float gaussian_mix_pdf(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float isqr_cov[M][N][N], const complex float x[N]);
extern void gaussian_mix_sample(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float sqr_cov[M][N][N], complex float x[N]);
extern void gaussian_score(int N, const complex float m[N], const complex float isqr_cov[N][N],
		const complex float x[N], complex float sc[N]);
extern void gaussian_mix_score(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float isqr_cov[M][N][N], const complex float x[N], complex float sc[N]);
extern void gaussian_multiply(int N, complex float m[N], complex float isqr_cov[N][N],
		const complex float m1[N], const complex float isqr_cov1[N][N],
		const complex float m2[N], const complex float isqr_cov2[N][N]);
extern void gaussian_convolve(int N, complex float m[N], complex float sqr_cov[N][N],
		const complex float m1[N], const complex float sqr_cov1[N][N],
		const complex float m2[N], const complex float sqr_cov2[N][N]);

