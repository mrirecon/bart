
#ifndef SNLOP_H
#define SNLOP_H

#include "misc/types.h"
#include "misc/shrdptr.h"
#include "misc/debug.h"

struct nlop_s;
struct snlop_s;
struct nlop_arg_s;

typedef struct nlop_arg_s* (*nlop_arg_reshape_f)(const struct nlop_arg_s* arg, long N, const long dims[N]);
typedef struct nlop_arg_s* (*nlop_arg_dup_f)(const struct nlop_arg_s* a, const struct nlop_arg_s* b);
typedef struct nlop_arg_s* (*nlop_arg_stack_f)(const struct nlop_arg_s* a, const struct nlop_arg_s* b, int stack_dim, _Bool out);
typedef void (*nlop_arg_del_f)(const struct nlop_arg_s* a);

struct nlop_arg_s {

    struct shared_obj_s sptr;
    TYPEID* TYPEID;

    nlop_arg_reshape_f reshape;
    nlop_arg_dup_f dup;
    nlop_arg_stack_f stack;
    nlop_arg_del_f del;

    struct snlop_s* x;
    const char* name;
    _Bool fixed_name;
};

typedef struct nlop_arg_s* arg_t;
typedef struct snlop_s* snlop_t;

extern struct nlop_arg_s* arg_ref(struct nlop_arg_s* x);
extern const struct nlop_arg_s* arg_unref(const struct nlop_arg_s* x);
extern int arg_ref_count(const struct nlop_arg_s* x);
extern void arg_init(arg_t arg);

extern void snlop_free(const struct snlop_s* x);

extern int snlop_nr_iargs(snlop_t snlop);
extern int snlop_nr_oargs(snlop_t snlop);
extern int snlop_nr_targs(snlop_t snlop);

extern arg_t snlop_get_iarg(snlop_t snlop, int i);
extern arg_t snlop_get_oarg(snlop_t snlop, int i);
extern arg_t snlop_get_targ(snlop_t snlop, int i);

extern _Bool arg_is_input(arg_t arg);
extern _Bool arg_is_output(arg_t arg);
extern void arg_set_name(arg_t arg, const char* name);
extern void arg_set_name_F(arg_t arg, const char* name);
void snlop_replace_iarg(arg_t narg, arg_t oarg);
void snlop_replace_oarg(arg_t narg, arg_t oarg);

extern snlop_t snlop_from_nlop_F(const struct nlop_s* nlop);

extern arg_t snlop_input(int N, const long dims[N], const char* name);
extern arg_t snlop_const(int N, const long dims[N], const _Complex float* data, const char* name);
extern arg_t snlop_scalar(_Complex float val);
extern void add_to_targs(arg_t arg);

extern void snlop_chain(int N, arg_t oargs[N], arg_t iargs[N], _Bool keep);
extern arg_t snlop_append_nlop_generic_F(int N, arg_t oargs[N], const struct nlop_s* nlop, _Bool keep);
extern arg_t snlop_append_nlop_F(arg_t oarg, const struct nlop_s* nlop, _Bool keep);
extern arg_t snlop_prepend_nlop_generic_F(int N, arg_t oargs[N], const struct nlop_s* nlop);
extern arg_t snlop_prepend_nlop_F(arg_t oarg, const struct nlop_s* nlop);

extern arg_t snlop_clone_arg(arg_t arg);
extern void snlop_del_arg(arg_t arg);

extern const struct iovec_s* arg_get_iov(arg_t arg);
extern const struct iovec_s* arg_get_iov_in(arg_t arg);
extern const struct iovec_s* arg_get_iov_out(arg_t arg);
extern const struct nlop_s* nlop_from_snlop_F(snlop_t snlop, int OO, arg_t oargs[OO], int II, arg_t iargs[II]);

extern snlop_t snlop_from_arg(arg_t arg);

extern arg_t arg_reshape(arg_t arg, int N, const long dims[N]);
extern arg_t arg_reshape_in(arg_t arg, int N, const long dims[N]);
extern arg_t arg_reshape_out(arg_t arg, int N, const long dims[N]);

extern arg_t snlop_stack(arg_t a, arg_t b, int stack_dim);
extern arg_t snlop_stack_F(arg_t a, arg_t b, int stack_dim);
extern arg_t snlop_stack_in(arg_t a, arg_t b, int stack_dim);
extern arg_t snlop_dup(arg_t a, arg_t b);

extern _Bool snlop_check(snlop_t snlop);
extern _Bool arg_check(arg_t arg);

extern void snlop_debug(enum debug_levels dl, struct snlop_s* x);

extern struct list_s* snlop_get_oargs(int N, arg_t args[N]);
extern void snlop_prune_oargs(arg_t oarg, struct list_s* keep_args);

extern void snlop_export_graph(snlop_t snlop, const char* path);

#endif