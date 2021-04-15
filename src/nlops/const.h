struct nlop_s;
extern struct nlop_s* nlop_const_create2(int N, const long dims[N], const long strs[N], _Bool copy, const _Complex float* in);
extern struct nlop_s* nlop_const_create(int N, const long dims[N], _Bool copy, const _Complex float* in);

struct nlop_s* nlop_set_input_const2(const struct nlop_s* a, int i, int N, const long dims[N], const long strs[N], _Bool copy, const _Complex float* in);
struct nlop_s* nlop_set_input_const(const struct nlop_s* a, int i, int N, const long dims[N], _Bool copy, const _Complex float* in);
struct nlop_s* nlop_set_input_const_F2(const struct nlop_s* a, int i, int N, const long dims[N], const long strs[N], _Bool copy, const _Complex float* in);
struct nlop_s* nlop_set_input_const_F(const struct nlop_s* a, int i, int N, const long dims[N], _Bool copy, const _Complex float* in);

extern struct nlop_s* nlop_del_out_create(int N, const long dims[N]);
extern struct nlop_s* nlop_del_out(const struct nlop_s* a, int o);
extern struct nlop_s* nlop_del_out_F(const struct nlop_s* a, int o);
