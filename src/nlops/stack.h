
struct nlop_s;

extern struct nlop_s* nlop_stack_generic_create(int II, int N, const long odims[N], const long idims[II][N], int stack_dim);
extern struct nlop_s* nlop_stack_create(int N, const long odims[__VLA(N)], const long idims1[__VLA(N)], const long idims2[__VLA(N)], int stack_dim);

extern struct nlop_s* nlop_destack_generic_create(int OO, int N, const long odims[OO][N], const long idims[N], int stack_dim);
extern struct nlop_s* nlop_destack_create(int N, const long odims1[__VLA(N)], const long odims2[__VLA(N)], const long idims[__VLA(N)], int stack_dim);

extern const struct nlop_s* nlop_stack_container_create(int N, const struct nlop_s* nlops[__VLA(N)], int II, int in_stack_dim[__VLA(II)], int OO, int out_stack_dim[__VLA(OO)]);
extern const struct nlop_s* nlop_stack_container_create_F(int N, const struct nlop_s* nlops[__VLA(N)], int II, int in_stack_dim[__VLA(II)], int OO, int out_stack_dim[__VLA(OO)]);

extern const struct nlop_s* nlop_flatten_stacked(const struct nlop_s* nlop);