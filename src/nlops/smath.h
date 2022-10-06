
struct arg_s;
struct nlop_s;
struct snlop_s;

typedef struct nlop_arg_s* arg_t;
typedef struct snlop_s* snlop_t;

extern arg_t snlop_abs(arg_t arg);
extern arg_t snlop_abs_F(arg_t arg);

extern arg_t snlop_exp(arg_t arg);
extern arg_t snlop_exp_F(arg_t arg);
extern arg_t snlop_log(arg_t arg);
extern arg_t snlop_log_F(arg_t arg);

extern arg_t snlop_cos(arg_t arg);
extern arg_t snlop_cos_F(arg_t arg);
extern arg_t snlop_sin(arg_t arg);
extern arg_t snlop_sin_F(arg_t arg);

extern arg_t snlop_cosh(arg_t arg);
extern arg_t snlop_cosh_F(arg_t arg);
extern arg_t snlop_sinh(arg_t arg);
extern arg_t snlop_sinh_F(arg_t arg);

extern arg_t snlop_real(arg_t arg);
extern arg_t snlop_real_F(arg_t arg);
extern arg_t snlop_conj(arg_t arg);
extern arg_t snlop_conj_F(arg_t arg);
extern arg_t snlop_sqrt(arg_t arg);
extern arg_t snlop_sqrt_F(arg_t arg);
extern arg_t snlop_inv(arg_t arg);
extern arg_t snlop_inv_F(arg_t arg);
extern arg_t snlop_spow(arg_t arg, _Complex float pow);
extern arg_t snlop_spow_F(arg_t arg, _Complex float pow);
extern arg_t snlop_scale(arg_t arg, _Complex float scale);
extern arg_t snlop_scale_F(arg_t arg, _Complex float scale);

extern arg_t snlop_cdiag(arg_t arg, int N, const long dims[N], const _Complex float* diag);
extern arg_t snlop_cdiag_F(arg_t arg, int N, const long dims[N], const _Complex float* diag);
extern arg_t snlop_fmac(arg_t arg, int N, const long dims[N], const _Complex float* ten, unsigned long oflags);
extern arg_t snlop_fmac_F(arg_t arg, int N, const long dims[N], const _Complex float* ten, unsigned long oflags);

extern arg_t snlop_stack(arg_t a, arg_t b, int stack_dim);
extern arg_t snlop_stack_F(arg_t a, arg_t b, int stack_dim);

extern arg_t snlop_mul(arg_t a, arg_t b, unsigned long flags);
extern arg_t snlop_mul_F(arg_t a, arg_t b, unsigned long flags);
extern arg_t snlop_div(arg_t a, arg_t b, unsigned long flags);
extern arg_t snlop_div_F(arg_t a, arg_t b, unsigned long flags);
extern arg_t snlop_mul_simple(arg_t a, arg_t b);
extern arg_t snlop_div_simple(arg_t a, arg_t b);

extern arg_t snlop_axpbz(arg_t a, arg_t b, _Complex float sa, _Complex float sb);
extern arg_t snlop_axpbz_F(arg_t a, arg_t b, _Complex float sa, _Complex float sb);

extern arg_t snlop_add(arg_t a, arg_t b);
extern arg_t snlop_add_F(arg_t a, arg_t b);
extern arg_t snlop_sub(arg_t a, arg_t b);
extern arg_t snlop_sub_F(arg_t a, arg_t b);

extern arg_t snlop_dump(arg_t arg, const char* name, _Bool frw, _Bool der, _Bool adj);