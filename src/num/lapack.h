
#include <complex.h>
#include <stdbool.h>

extern void lapack_eig(long N, float eigenval[N], complex float matrix[N][N]);
extern void lapack_geig(long N, float eigenval[N], complex float A[N][N], complex float B[N][N]);
extern void lapack_svd(long M, long N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], complex float A[N][M]);
extern void lapack_svd_econ(long M, long N,
		     complex float U[(N > M) ? M : N][M],
		     complex float VH[N][(N > M) ? M : N],
		     float S[(N > M) ? M : N],
		     complex float A[N][M]);

extern void lapack_qr_econ(long M, long N,
		    complex float R[N][(N > M) ? M : N],
		    complex float A[N][M]);

extern void lapack_eig_double(long N, double eigenval[N], complex double matrix[N][N]);
extern void lapack_svd_double(long M, long N, complex double U[M][M], complex double VH[N][N], double S[(N > M) ? M : N], complex double A[N][M]);
extern void lapack_matrix_multiply(long M, long N, long K, complex float C[M][N], const complex float A[M][K], const complex float B[K][N]);

extern void lapack_cholesky(long N, complex float A[N][N]);
extern void lapack_cholesky_lower(long N, complex float A[N][N]);

extern void lapack_trimat_inverse(long N, complex float A[N][N]);
extern void lapack_trimat_inverse_lower(long N, complex float A[N][N]);
extern void lapack_trimat_solve(long N, long M, complex float A[N][N], complex float B[M][N], bool upper);

extern void lapack_schur(long N, complex float W[N], complex float VS[N][N], complex float A[N][N]);
extern void lapack_schur_double(long N, complex double W[N], complex double VS[N][N], complex double A[N][N]);

extern void lapack_sylvester(long N, long M, float* scale, complex float A[N][N], complex float B[M][M], complex float C[M][N]);

extern void lapack_cinverse_UL(long N, complex float A[N][N]);
extern void lapack_sinverse_UL(long N, float A[N][N]);

extern void lapack_solve_real(long N, float A[N][N], float B[N]);


