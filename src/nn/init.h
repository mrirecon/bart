#ifndef INITIALIZER_H
#define INITIALIZER_H

struct initializer_s;
typedef void (*initializer_f)(const struct initializer_s* conf, long N, const long dims[N], _Complex float* weights);

extern void initializer_apply(const struct initializer_s* conf, long N, const long dims[N], _Complex float* weights);
extern void initializer_free(const struct initializer_s* conf);
extern const struct initializer_s* initializer_clone(const struct initializer_s* x);

extern const struct initializer_s* init_reshape_create(unsigned int N, const long dims[N], const struct initializer_s* init);
extern const struct initializer_s* init_stack_create(unsigned int N, int stack_dim, const long dimsa[N], const struct initializer_s* inita, const long dimsb[N], const struct initializer_s* initb);
extern const struct initializer_s* init_dup_create(const struct initializer_s* inita, const struct initializer_s* initb);

extern unsigned long in_flag_conv(_Bool c1);
extern unsigned long out_flag_conv(_Bool c1);

extern unsigned long in_flag_conv_generic(int N, unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag);
extern unsigned long out_flag_conv_generic(int N, unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag);

extern const struct initializer_s* init_const_create(_Complex float val);
extern const struct initializer_s* init_xavier_create(unsigned long in_flags, unsigned long out_flags, _Bool real, _Bool uniform);
extern const struct initializer_s* init_kaiming_create(unsigned long in_flags, _Bool real, _Bool uniform, float leaky_val);
extern const struct initializer_s* init_array_create(int N, const long dims[N], const _Complex float* dat);

extern const struct initializer_s* init_std_normal_create(_Bool real, float scale, float mean);
extern const struct initializer_s* init_uniform_create(_Bool real, float scale, float mean);

extern const struct initializer_s* init_linspace_create(unsigned int dim, _Complex float min_val, _Complex float max_val, _Bool max_inc);

#endif