
struct nlop_s;
extern float nlop_test_derivative(const struct nlop_s* op);

extern float nlop_test_adj_derivatives(const struct nlop_s* op, _Bool real);
extern float nlop_test_derivatives(const struct nlop_s* op);
extern _Bool nlop_test_derivatives_reduce(const struct nlop_s* op, int iter_max, int reduce_target, float val_target);

extern _Bool compare_nlops(const struct nlop_s* nlop1, const struct nlop_s* nlop2, _Bool shape, _Bool der, _Bool adj, float tol);
extern float compare_gpu(const struct nlop_s* cpu_op, const struct nlop_s* gpu_op);
