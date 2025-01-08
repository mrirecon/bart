
extern const struct nn_s* nn_score_to_expectation(const struct nn_s* score);
extern const struct nn_s* nn_expectation_to_score(const struct nn_s* score);

extern const struct nlop_s* nlop_score_to_expectation(const struct nlop_s* score);
extern const struct nlop_s* nlop_expectation_to_score(const struct nlop_s* score);

extern const struct nn_s* nn_denoise_precond_edm(const struct nn_s* network, float sigma_min, float sigma_max, float sigma_data, _Bool ambient);

extern const struct nn_s* nn_denoise_loss_VE(const struct nn_s* network, float sigma_min, float sigma_max, float sigma_data);
extern const struct nn_s* nn_denoise_loss_EDM(const struct nn_s* network, float sigma_min, float sigma_max, float sigma_data);
