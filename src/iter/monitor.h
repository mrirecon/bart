
#ifndef __ITER_MONITOR_H
#define __ITER_MONITOR_H


struct iter_monitor_s;
struct vec_iter_s;
struct typeid_s;
struct iter_history_s { const struct typeid_s* TYPEID; };

typedef struct iter_history_s iter_history_t;

typedef void (*iter_monitor_fun_t)(struct iter_monitor_s* data, const struct vec_iter_s* ops, const float* x);
typedef void (*iter_history_fun_t)(struct iter_monitor_s* data, const struct iter_history_s*);

struct iter_monitor_s {

	const struct typeid_s* TYPEID;
	iter_monitor_fun_t fun;
	iter_history_fun_t record;

	double obj;
	double err;
};

typedef struct iter_monitor_s iter_monitor_t;

extern void iter_monitor(struct iter_monitor_s* monitor, const struct vec_iter_s* ops, const float* x);
extern void iter_history(struct iter_monitor_s* monitor, const struct iter_history_s*);

extern struct iter_monitor_s* create_monitor(long N, const float* image_truth,
		void* data, float (*object)(const void* data, const float* x));


#endif // __ITER_MONITOR_H


