#include <stddef.h>

#include "misc/cppwrap.h"

enum delayed_md_fun_type {
	delayed_op_type_z3op,
	delayed_op_type_3op,
	delayed_op_type_z3opd,
	delayed_op_type_3opd,
	delayed_op_type_z2op,
	delayed_op_type_2op,
	delayed_op_type_z2opd,
	delayed_op_type_2opd,
	delayed_op_type_z2opf,
	delayed_op_type_2opf,
};

struct list_s;
struct vptr_fun_data_s;
typedef void (*vptr_fun_t)(struct vptr_fun_data_s* data, int N, int D, const long* dims[__VLA(N)], const long* strs[__VLA(N)], void* args[__VLA(N)]);

extern void exec_vptr_fun_delayed(vptr_fun_t fun, struct vptr_fun_data_s* data, int N, int D, unsigned long lflags, unsigned long wflags, unsigned long rflags, const long* dims[__VLA(N)], const long* strs[__VLA(N)], void* ptr[__VLA(N)], size_t sizes[__VLA(N)], _Bool resolve);

extern void debug_delayed_queue(int dl, struct list_s* ops_queue, _Bool nested);
extern void delayed_compute(const void* ptr);

extern _Bool is_delayed(const void* ptr);
extern void delayed_alloc(const void* ptr, int N, const long dims[__VLA(N)], size_t size);
extern void delayed_free(const void* ptr, int N, const long dims[__VLA(N)], size_t size);

extern _Bool delayed_queue_copy(int D, const long dim[__VLA(D)], const long ostr[__VLA(D)], void* optr, const long istr[__VLA(D)], const void* iptr, size_t size);
extern _Bool delayed_queue_circ_shift(int D, const long dimensions[D], const long center[D], const long str1[D], void* dst, const long str2[D], const void* src, size_t size);
extern _Bool delayed_queue_clear(int D, const long dim[__VLA(D)], const long str[__VLA(D)], void* ptr, size_t size);
extern _Bool delayed_queue_make_op(enum delayed_md_fun_type type, size_t offset, int D, const long dim[__VLA(D)], int N, const long* strs[__VLA(N)], const void* ptr[__VLA(N)], const size_t sizes[__VLA(N)]);

//extern for testing, dont use!

struct queue_s;
struct delayed_op_s;

extern struct queue_s* get_global_queue(void);
extern void release_global_queue(struct queue_s* queue);
extern void delayed_queue_exec(struct queue_s* queue);
extern void delayed_compute_debug(const char* name);

extern void queue_set_compute(struct queue_s*, _Bool compute);
extern struct list_s* get_delayed_op_list(struct queue_s* queue);
extern void delayed_optimize_queue(struct list_s* ops_queue);
extern void delayed_optimize_queue_looping(struct list_s* ops_queue);

extern _Bool delayed_op_is_alloc(const struct delayed_op_s* op);
extern _Bool delayed_op_is_free(const struct delayed_op_s* op);
extern _Bool delayed_op_is_copy(const struct delayed_op_s* op);
extern _Bool delayed_op_is_clear(const struct delayed_op_s* op);
extern _Bool delayed_op_is_chain(const struct delayed_op_s* op);

extern void debug_mpeak_queue(int dl, struct list_s* ops_queue, _Bool node);
extern long compute_mpeak(struct list_s* ops_queue, _Bool node);
extern long compute_mchange(struct list_s* ops_queue, _Bool node);


#include "misc/cppwrap.h"

