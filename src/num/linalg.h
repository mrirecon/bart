
#include <complex.h>
#include <stdbool.h>

extern void mat_identity(int A, int B, complex float x[A][B]);
extern void mat_zero(int A, int B, complex float x[A][B]);
extern void mat_gaussian(int A, int B, complex float x[A][B]);
extern void mat_mul(int A, int B, int C, complex float x[A][C], const complex float y[A][B], const complex float z[B][C]);

extern void matf_mul(int A, int B, int C, float x[A][C], const float y[A][B], const float z[B][C]);

#define MVLA(x) __restrict__ (A)
extern void mat_muladd(int A, int B, int C, complex float x[MVLA(A)][C], const complex float y[MVLA(A)][B], const complex float z[MVLA(B)][C]);
extern void mat_add(int A, int B, complex float x[A][B], const complex float y[A][B], const complex float z[A][B]);
extern void mat_transpose(int A, int B, complex float dst[B][A], const complex float src[A][B]);
extern void matf_transpose(int A, int B, float dst[B][A], const float src[A][B]);
extern void mat_adjoint(int A, int B, complex float dst[B][A], const complex float src[A][B]);
extern void mat_conj(int A, int B, complex float dst[A][B], const complex float src[A][B]);
extern void mat_copy(int A, int B, complex float dst[A][B], const complex float src[A][B]);
extern void matf_copy(int N, int M, float out[N][M], const float in[N][M]);
extern bool mat_inverse(int N, complex float dst[N][N], const complex float src[N][N]);
extern void mat_pinv(int A, int B, complex float out[B][A], const complex float in[A][B]);
extern void mat_pinv_left(int A, int B, complex float out[B][A], const complex float in[A][B]);
extern void mat_pinv_right(int A, int B, complex float out[B][A], const complex float in[A][B]);
extern void mat_svd_recov(int A, int B, complex float out[A][B], const complex float U[A][A], const complex float VH[B][B], const float S[A]);
extern void mat_svd(int A, int B, complex float U[A][A], complex float VH[B][B], float S[A], const complex float in[A][B]);
extern void mat_pinv_svd(int A, int B, complex float out[B][A], const complex float in[A][B]);
extern void mat_schur_recov(int A, complex float out[A][A], const complex float T[A][A], const complex float Z[A][A]);
extern void mat_schur(int A, complex float T[A][A], complex float Z[A][A], const complex float in[A][A]);
extern void mat_ceig_double(int A, complex double EV[A], const complex double in[A][A]);
extern void mat_eig_double(int A, double EV[A], const double in[A][A]);
extern complex float mat_det(int N, const complex float mat[N][N]);
extern void mat_vecmul(int A, int B, complex float out[A], const complex float mat[A][B], const complex float in[B]);
extern void matf_vecmul(int A, int B, float out[A], const float mat[A][B], const float in[B]);
extern void matf_solve(int N, float x[N], const float m[N][N], const float y[N]);

extern void mat_kron(int A, int B, int C, int D,
		complex float out[A * C][B * D], const complex float in1[A][B], const complex float in2[C][D]);
extern void mat_vec(int A, int B, complex float out[A * B], const complex float in[A][B]);
extern void vec_mat(int A, int B, complex float out[A][B], const complex float in[A * B]);
extern void vec_zero(int N, complex float x[N]);
extern void vec_copy(int N, complex float x[N], const complex float y[N]);
extern void vecf_copy(int N, float x[N], const float y[N]);
extern float vecf_sdot(int N, const float a[N], const float b[N]);
extern float vecf_norm(int N, const float x[N]);

extern double vec3d_sdot(const double* x, const double* y);
extern double vec3d_norm(const double* x);
extern void vec3d_saxpy(double* o, const double* x, const double a, const double* y);
extern void vec3d_set(double* x, const double v);
extern void vec3d_cp(double* o, const double* v0, const double* v1);
extern void vec3d_rotax(double* o, const double theta, const double* ax, const double* x);
extern void vec3d_copy(double* o, const double* x);
extern double vec3d_angle(const double* x, const double* y);

// extern complex double vec_dot(int N, const complex float x[N], const complex float y[N]);
extern complex float vec_dot(int N, const complex float x[N], const complex float y[N]);
extern void vec_saxpy(int N, complex float x[N], complex float alpha, const complex float y[N]);
extern void vecf_saxpy(int N, float x[N], float alpha, const float y[N]);
extern void gram_matrix(int N, complex float cov[N][N], int L, const complex float data[N][L]);
extern void gram_schmidt(int M, int N, float val[M], complex float vecs[M][N]);
extern void gram_matrix2(int N, complex float cov[N * (N + 1) / 2], int L, const complex float data[N][L]);
extern void pack_tri_matrix(int N, complex float cov[N * (N + 1) / 2], const complex float m[N][N]);
extern void unpack_tri_matrix(int N, complex float m[N][N], const complex float cov[N * (N + 1) / 2]);
extern void solve_tri_matrix(int A, int B, complex float M[A][A], complex float N[A][B], bool upper);
extern void solve_tri_matrix_vec(int A, complex float M[A][A], complex float N[A], bool upper);
extern void solve_tri_matrix_sylvester(int A, int B, float* scale, complex float M[A][A], complex float N[B][B], complex float C[A][B]);
extern void sqrtm_tri_matrix(int N, int blocksize, complex float out[N][N], const complex float in[N][N]);
extern float mat_onenorm_power(int N, int order, complex float A[N][N]);
extern void logm_tri_matrix(int N, complex float out[N][N], const complex float in[N][N]);
extern void mat_logm(int N, complex float out[N][N], complex float in[N][N]);
extern void orthiter_noinit(int M, int N, int iter, float vals[M], complex float out[M][N], const complex float matrix[N][N]);
extern void orthiter(int M, int N, int iter, float vals[M], complex float out[M][N], const complex float matrix[N][N]);
extern void cholesky(int N, complex float A[N][N]);
extern void cholesky_solve(int N, complex float x[N], const complex float L[N][N], const complex float b[N]);
extern void cholesky_double(int N, complex double A[N][N]);
extern void cholesky_solve_double(int N, complex double x[N], const complex double L[N][N], const complex double b[N]);
extern complex float vec_mean(long D, const complex float src[D]);
extern void vec_axpy(long N, complex float x[N], complex float alpha, const complex float y[N]);
extern void vec_sadd(long D, complex float alpha, complex float dst[D], const complex float src[D]);
extern void thomas_algorithm(int N, complex float f[N], const complex float A[N][3], const complex float d[N]);

extern void mat_band_reorder(int A, int B, double mat[A][A], double band[B][A], bool upper);

