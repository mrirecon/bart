
struct nlop_s;
extern struct nlop_s* nlop_stack_create(int N, const long odims[__VLA(N)], const long idims1[__VLA(N)], const long idims2[__VLA(N)], unsigned long stack_dim);
extern struct nlop_s* nlop_destack_create(int N, const long odims1[__VLA(N)], const long odims2[__VLA(N)], const long idims[__VLA(N)], unsigned long stack_dim);

