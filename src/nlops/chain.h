


struct nlop_s;

extern struct nlop_s* nlop_chain(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_chain_FF(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_chain2(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_combine(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_link(const struct nlop_s* x, int oo, int ii);
extern struct nlop_s* nlop_permute_inputs(const struct nlop_s* x, int I2, const int perm[I2]);


