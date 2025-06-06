
extern const struct nlop_s* nlop_mse_create(int N, const long dims[__VLA(N)], unsigned long mean_dims);
extern const struct nlop_s* nlop_mse_scaled_create(int N, const long dims[__VLA(N)], unsigned long mean_dims);
extern const struct nlop_s* nlop_mpsnr_create(int N, const long dims[__VLA(N)], unsigned long mean_dims);
extern const struct nlop_s* nlop_mssim_create(int N, const long dims[__VLA(N)], const long kdims[__VLA(N)], unsigned long conv_dims);
extern const struct nlop_s* nlop_patched_cross_correlation_create(int N, const long dims[__VLA(N)], const long kdims[__VLA(N)], unsigned long conv_dims, float epsilon);
extern const struct nlop_s* nlop_cce_create(int N, const long dims[__VLA(N)], unsigned long batch_flag);
extern const struct nlop_s* nlop_weighted_cce_create(int N, const long dims[__VLA(N)], unsigned long batch_flag);
extern const struct nlop_s* nlop_accuracy_create(int N, const long dims[__VLA(N)], int class_index);

extern const struct nlop_s* nlop_nmse_create(int N, const long dims[N], unsigned long batch_flags);
extern const struct nlop_s* nlop_nrmse_create(int N, const long dims[N], unsigned long batch_flags);

extern const struct nlop_s* nlop_dice_generic_create(int N, const long dims[N], unsigned long label_flag, unsigned long independent_flag, float weighting_exponent, _Bool square_denominator);
extern const struct nlop_s* nlop_dice_create(int N, const long dims[N], unsigned long label_flag, unsigned long mean_flag, float weighting_exponent, _Bool square_denominator);
