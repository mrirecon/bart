
#ifndef __MONITOR_ITER6_H
#define __MONITOR_ITER6_H


struct monitor_iter6_s;
struct typeid_s;
struct nlop_s;

struct monitor_iter6_s;
struct monitor_value_s;

extern void monitor_iter6_free(const struct monitor_iter6_s* monitor);
extern struct monitor_iter6_s* monitor_iter6_create(_Bool progressbar, _Bool record, int M, const struct monitor_value_s* val_monitors[M]);
extern void monitor6_average_objective(struct monitor_iter6_s* monitor);

struct monitor_value_s* monitor_iter6_nlop_create(const struct nlop_s* nlop, _Bool eval_each_batch, unsigned int N, const char* print_names[N]);

typedef _Complex float (*monitor_iter6_value_by_function_t)(long NI, const float* args[NI]);

struct monitor_value_s* monitor_iter6_function_create(monitor_iter6_value_by_function_t fun, _Bool eval_each_batch, const char* print_name);

extern void monitor_iter6_dump_record(struct monitor_iter6_s* _monitor, const char* filename);


#endif // __MONITOR_ITER6_H
