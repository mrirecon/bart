extern const struct operator_s* operator_clip_create(unsigned int N, const long dims[__VLA(N)], float clipnorm, float clipval);
extern const struct operator_p_s* operator_adadelta_update_create(unsigned int N, const long dims[__VLA(N)], float rho, float epsilon);
extern const struct operator_p_s* operator_adam_update_create(unsigned int N, const long dims[__VLA(N)], float beta1, float beta2, float epsilon, long reset_mod);
extern const struct operator_p_s* operator_sgd_update_create(unsigned int N, const long dims[__VLA(N)]);