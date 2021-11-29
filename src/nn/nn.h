
#ifndef NN_H
#define NN_H

#include "misc/debug.h"

#include "iter/italgos.h"
#include "nlops/nlop.h"

struct operator_p_s;
struct initializer_s;

/*
In this file, we define the struct nn_s and its type nn_t.

This struct is a wrapper around a non-linear operator which should simplify the composition of complex neural networks.
While inputs and outputs of non-linear operators are indexed with a positive index, the nn_t type supports both indexing using name strings and nummeric indexing.
Note that named arguments are not counted for numeric indexing, but a named input might be between to numeric indices.
Negative indices can be used to start counting from the last index (this is not possible for nlops).

An example how the consecutive inputs of a nn_t might be accessed is:
[0, 1, "weight1", "bias27", 2, "pattern", -3, 4, "kspace" -1]

Functions working with a specific input/output are usually passed an integer and a string. The integer value is only used for indexing if the string points to NULL, else the string is used.
To avoid confusions, the integer must be 0 or -1 in case the string is used for indexing.
Examples

- (0, NULL) will access input 0
- (2, NULL) will access input 4
- (0, "pattern") will access input 4
- (3, "pattern") will produce an error
- (0, "pAttern") will produce an error as the string is not found

In the definition of the functions in this file, we use the term "index" for the numeric indices of a nn_t type and the term "arg_index" for the indices of the correponding non-linear operator.


Moreover, the nn_t type can store an initializer for each input and the types of inputs/outputs used for optimization.

*/
struct nn_s {

	const struct nlop_s* nlop;

	const char** out_names;
	const char**  in_names;

	const struct initializer_s** initializers;

	const struct operator_p_s** prox_ops;

	bool* dup; // can the input be duplicated, i.e. can it be used for shared weights

	enum IN_TYPE* in_types;
	enum OUT_TYPE* out_types;
};

typedef const struct nn_s* nn_t;

extern nn_t nn_from_nlop(const struct nlop_s* op);
extern nn_t nn_from_nlop_F(const struct nlop_s* op);
extern void nn_free(nn_t op);

extern const struct nlop_s* nn_get_nlop(nn_t op);

extern nn_t nn_clone(nn_t op);
extern void nn_clone_arg_i_from_i(nn_t nn1, int i1, nn_t nn2, int i2);
extern void nn_clone_arg_o_from_o(nn_t nn1, int o1, nn_t nn2, int o2);
extern void nn_clone_args(nn_t dst, nn_t src);

extern int nn_get_nr_named_in_args(nn_t op);
extern int nn_get_nr_named_out_args(nn_t op);
extern int nn_get_nr_unnamed_in_args(nn_t op);
extern int nn_get_nr_unnamed_out_args(nn_t op);
extern int nn_get_nr_in_args(nn_t op);
extern int nn_get_nr_out_args(nn_t op);

extern int nn_get_out_arg_index(nn_t op, int o, const char* oname);
extern int nn_get_in_arg_index(nn_t op, int i, const char* iname);

extern _Bool nn_is_num_in_index(nn_t op, int i);
extern _Bool nn_is_num_out_index(nn_t op, int o);

extern void nn_get_in_args_names(nn_t op, int nII, const char* inames[nII], _Bool copy);
extern void nn_get_out_args_names(nn_t op, int nOO, const char* onames[nOO], _Bool copy);

extern const char* nn_get_in_name_from_arg_index(nn_t op, int i, _Bool clone);
extern const char* nn_get_out_name_from_arg_index(nn_t op, int o, _Bool clone);

extern int nn_get_in_index_from_arg_index(nn_t op, int i);
extern int nn_get_out_index_from_arg_index(nn_t op, int o);

extern _Bool nn_is_name_in_in_args(nn_t op, const char* name);
extern _Bool nn_is_name_in_out_args(nn_t op, const char* name);

extern nn_t nn_set_input_name_F(nn_t op, int i, const char* iname);
extern nn_t nn_set_output_name_F(nn_t op, int o, const char* oname);
extern nn_t nn_rename_input_F(nn_t op, const char* nname, const char* oname);
extern nn_t nn_rename_output_F(nn_t op, const char* nname, const char* oname);
extern nn_t nn_unset_input_name_F(nn_t op, const char* iname);
extern nn_t nn_unset_output_name_F(nn_t op, const char* oname);

extern nn_t nn_set_initializer_F(nn_t op, int i, const char* iname, const struct initializer_s* ini);
extern nn_t nn_set_in_type_F(nn_t op, int i, const char* iname, enum IN_TYPE in_type);
extern nn_t nn_set_out_type_F(nn_t op, int o, const char* oname, enum OUT_TYPE out_type);

extern nn_t nn_set_prox_op_F(nn_t op, int i, const char* iname, const struct operator_p_s* opp);
extern const struct operator_p_s* nn_get_prox_op(nn_t op, int i, const char* iname);
extern const struct operator_p_s* nn_get_prox_op_arg_index(nn_t op, int i);
extern void nn_get_prox_ops(nn_t op, int N, const struct operator_p_s* prox_ops[N]);

extern nn_t nn_set_dup_F(nn_t op, int i, const char* iname, bool dup);
extern _Bool nn_get_dup(nn_t op, int i, const char* iname);

extern const char** nn_get_out_names(nn_t op);
extern const char** nn_get_in_names(nn_t op);
extern void nn_get_in_names_copy(int N, const char* names[N], nn_t op);
extern void nn_get_out_names_copy(int N, const char* names[N], nn_t op);

extern int nn_get_nr_weights(nn_t op);
extern void nn_get_in_types(nn_t op, int N, enum IN_TYPE in_types[N]);
extern void nn_get_out_types(nn_t op, int N, enum OUT_TYPE out_types[N]);

extern const struct iovec_s* nn_generic_domain(nn_t op, int i, const char* iname);
extern const struct iovec_s* nn_generic_codomain(nn_t op, int o, const char* oname);

extern nn_t nn_checkpoint_F(nn_t op, _Bool der_once, _Bool clear_mem);

extern void nn_debug(enum debug_levels dl, nn_t x);
extern void nn_export_graph(const char* filename, nn_t op);

#endif