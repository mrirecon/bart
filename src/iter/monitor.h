


struct iter_monitor_s;
struct vec_iter_s;
struct typeid_s;
typedef void (*iter_monitor_fun_t)(struct iter_monitor_s* data, const struct vec_iter_s* ops, const float* x);
struct iter_monitor_s { const struct typeid_s* TYPEID; iter_monitor_fun_t fun; };

typedef struct iter_monitor_s iter_monitor_t;

extern void iter_monitor(struct iter_monitor_s* monitor, const struct vec_iter_s* ops, const float* x);

extern struct iter_monitor_s* create_monitor(long N, const float* image_truth,
		void* data, float (*object)(const void* data, const float* x));

