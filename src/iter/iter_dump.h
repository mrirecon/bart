
#ifndef __ITER_DUMP_H
#define __ITER_DUMP_H

struct typeid_s;
struct nlop_s;

struct iter_dump_s;
typedef void (*iter_dump_fun_t)(const struct iter_dump_s* data, long epoch, long NI, const float* x[NI]);
typedef void (*iter_dump_free_t)(const struct iter_dump_s* data);

struct iter_dump_s {

	const struct typeid_s* TYPEID;
	iter_dump_fun_t fun;
	iter_dump_free_t free;
	const char* base_filename;
};

const struct iter_dump_s* iter_dump_default_create(const char* base_filename, long save_mod, long NI, unsigned long save_flag, int D[NI], const long* dims[NI]);

void iter_dump(const struct iter_dump_s* data, long epoch, long NI, const float* x[NI]);
void iter_dump_free(const struct iter_dump_s* data);


#endif // __ITER_DUMP_H
