
struct operator_s;
extern double iter_power(int maxiter,
		const struct operator_s* normaleq_op,
		long size, float* u);

extern double estimate_maxeigenval(const struct operator_s* op);
extern double estimate_maxeigenval_gpu(const struct operator_s* op);

extern double estimate_maxeigenval_sameplace(const struct operator_s* op, int iterations, const void *ref);
