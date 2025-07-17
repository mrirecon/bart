struct nlop_s;
extern struct nlop_s* nlop_gmm_score_create(int N, const long score_dims[N], const long mean_dims[N], const _Complex float* mean, const long var_dims[N], const _Complex float* var, const long wgh_dims[N], const _Complex float* wgh);
