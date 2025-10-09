

struct operator_s;

extern void randomized_subspace_iteration_block(const struct operator_s* A, const struct operator_s* AH, int q, long M, long K, _Complex float Q[K][M]);
extern void randomized_svd_block(const struct operator_s* A, const struct operator_s* AH, int q, long M, long N, long K, long P, _Complex float U[K][M], _Complex float VH[N][K], float S[K]);
extern void randomized_eig_block(const struct operator_s* op, int q, long N, long K, long P, _Complex float U[K][N], float S[K]);

extern void randomized_svd_dense(int q, long M, long N, long K, long P, _Complex float U[K][M], _Complex float VH[N][K], float S[K], const _Complex float mat[N][M]);
extern void randomized_eig_dense(int q, long N, long K, long P, _Complex float U[K][N], float S[K], const _Complex float mat[N][N]);
