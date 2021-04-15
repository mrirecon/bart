
struct nlop_data_s;

extern const struct nlop_s* nlop_checkpoint_create(const struct nlop_s* nlop, _Bool der_once, _Bool clear_mem);
extern const struct nlop_s* nlop_checkpoint_create_F(const struct nlop_s* nlop, _Bool der_once, _Bool clear_mem);
extern void nlop_clear_derivatives_checkpoint(const struct nlop_data_s* _data, _Bool enforce);