
extern const struct nlop_s* nlop_znorm_create(int N, const long dims[N], unsigned long mean_dims);
extern const struct nlop_s* nlop_mse_create(int N, const long dims[N], unsigned long mean_dims);
extern const struct nlop_s* nlop_mse_scaled_create(int N, const long dims[N], unsigned long mean_dims);
extern const struct nlop_s* nlop_zasum_create(int N, const long dims[N], unsigned long mean_dims);
extern const struct nlop_s* nlop_z1norm_create(int N, const long dims[N], unsigned long mean_dims);
extern const struct nlop_s* nlop_mad_create(int N, const long dims[N], unsigned long mean_dims);
extern const struct nlop_s* nlop_nmse_create(int N, const long dims[N], unsigned long batch_flags);
extern const struct nlop_s* nlop_nrmse_create(int N, const long dims[N], unsigned long batch_flags);

