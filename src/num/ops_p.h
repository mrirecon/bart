

struct operator_data_s;
typedef struct operator_data_s operator_data_t;

typedef void (*operator_p_fun_t)(const operator_data_t* _data, float mu, _Complex float* _dst, const _Complex float* _src);
typedef void (*operator_del_t)(const operator_data_t* _data);


struct operator_s;
struct operator_p_s;

extern const struct operator_p_s* operator_p_create(unsigned int ON, const long out_dims[__VLA(ON)],
			unsigned int IN, const long in_dims[__VLA(IN)], operator_data_t* data,
			operator_p_fun_t apply, operator_del_t del);

extern const struct operator_p_s* operator_p_create2(unsigned int ON, const long out_dims[__VLA(ON)], const long out_strs[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)], const long in_strs[__VLA(IN)],
		operator_data_t* data, operator_p_fun_t apply, operator_del_t del);

extern void operator_p_free(const struct operator_p_s* x);
extern const struct operator_p_s* operator_p_ref(const struct operator_p_s* x);


extern const struct operator_p_s* operator_p_pre_chain(const struct operator_s* a, const struct operator_p_s* b);
extern const struct operator_p_s* operator_p_pst_chain(const struct operator_p_s* a, const struct operator_s* b);

extern const struct operator_s* operator_p_bind(const struct operator_p_s* op, float alpha);
extern const struct operator_p_s* operator_p_stack(int A, int B, const struct operator_p_s* a, const struct operator_p_s* b);

extern const struct operator_p_s* operator_p_scale(int N, const long dims[N]);


extern void operator_p_apply(const struct operator_p_s* op, float mu, unsigned int ON, const long odims[__VLA(ON)], _Complex float* dst, const long IN, const long idims[__VLA(IN)], const _Complex float* src);
extern void operator_p_apply2(const struct operator_p_s* op, float mu, unsigned int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)], _Complex float* dst, const long IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], const _Complex float* src);


extern void operator_p_apply_unchecked(const struct operator_p_s* op, float mu,  _Complex float* dst, const _Complex float* src);


// get functions
struct iovec_s;

extern const struct iovec_s* operator_p_domain(const struct operator_p_s* op);
extern const struct iovec_s* operator_p_codomain(const struct operator_p_s* op);

extern operator_data_t* operator_p_get_data(const struct operator_p_s* x);

extern const struct operator_s* operator_p_upcast(const struct operator_p_s* op);
extern const struct operator_p_s* operator_p_downcast(const struct operator_s* op);

extern const struct operator_p_s* operator_p_gpu_wrapper(const struct operator_p_s* op);


