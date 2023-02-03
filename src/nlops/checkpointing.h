
struct nlop_data_s;

extern const struct nlop_s* nlop_checkpoint_create(const struct nlop_s* nlop, _Bool der_once, _Bool clear_mem);
extern const struct nlop_s* nlop_checkpoint_create_F(const struct nlop_s* nlop, _Bool der_once, _Bool clear_mem);

extern const struct nlop_s* nlop_loop_generic_F(int N, const struct nlop_s* nlop, int II, int iloop_dim[__VLA(II)], int OO, int oloop_dim[__VLA(OO)]);
extern const struct nlop_s* nlop_loop_F(int N, const struct nlop_s* nlop, unsigned long dup_flag, int loop_dim);

extern _Bool nlop_is_checkpoint(const struct nlop_s* nlop);