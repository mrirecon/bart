
#ifndef NN_CHAIN_H
#define NN_CHAIN_H

#include "nn/nn.h"

extern nn_t nn_reshape_out(nn_t op, int o, const char* oname, int NO, const long odims[NO]);
extern nn_t nn_reshape_in(nn_t op, int i, const char* iname, int NI, const long idims[NI]);
extern nn_t nn_reshape_out_F(nn_t op, int o, const char* oname, int NO, const long odims[NO]);
extern nn_t nn_reshape_in_F(nn_t op, int i, const char* iname, int NI, const long idims[NI]);
extern nn_t nn_chain2(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname);
extern nn_t nn_chain2_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname);
extern nn_t nn_chain2_swap_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname);
extern nn_t nn_chain2_keep(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname);
extern nn_t nn_chain2_keep_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname);
extern nn_t nn_chain2_keep_swap_FF(nn_t a, int o, const char* oname, nn_t b, int i, const char* iname);
extern nn_t nn_combine(nn_t a, nn_t b);
extern nn_t nn_combine_FF(nn_t a, nn_t b);
extern nn_t nn_link(nn_t x, int oo, const char* ooname, int ii, const char* iiname);
extern nn_t nn_link_F(nn_t x, int oo, const char* ooname, int ii, const char* iiname);
extern nn_t nn_permute_inputs(nn_t x, int I2, const int perm[__VLA(I2)]);
extern nn_t nn_permute_outputs(nn_t x, int O2, const int perm[__VLA(O2)]);
extern nn_t nn_permute_inputs_F(nn_t x, int I2, const int perm[__VLA(I2)]);
extern nn_t nn_permute_outputs_F(nn_t x, int O2, const int perm[__VLA(O2)]);
extern nn_t nn_dup(nn_t x, int a, const char* aname, int b, const char* bname);
extern nn_t nn_dup_F(nn_t x, int a, const char* aname, int b, const char* bname);
extern nn_t nn_stack_inputs(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim);
extern nn_t nn_stack_inputs_F(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim);
extern nn_t nn_stack_outputs(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim);
extern nn_t nn_stack_outputs_F(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim);

extern nn_t nn_shift_input_index_F(nn_t x, int n, int o);
extern nn_t nn_shift_output_index_F(nn_t x, int n, int o);
extern nn_t nn_shift_input_F(nn_t x, int n, const char* nname, int o, const char* oname);
extern nn_t nn_shift_output_F(nn_t x, int n, const char* nname, int o, const char* oname);

extern nn_t nn_append_singleton_dim_in_F(nn_t op, int i, const char* iname);
extern nn_t nn_append_singleton_dim_out_F(nn_t op, int o, const char* oname);

extern nn_t nn_mark_dup_F(nn_t x, const char* name);
extern nn_t nn_mark_stack_input_F(nn_t x, const char* name);
extern nn_t nn_mark_stack_output_F(nn_t x, const char* name);

extern nn_t nn_stack_dup_by_name_F(nn_t x);
extern nn_t nn_sort_inputs_by_list_F(nn_t x, int N, const char* sorted_names[N]);
extern nn_t nn_sort_outputs_by_list_F(nn_t x, int N, const char* sorted_names[N]);
extern nn_t nn_sort_inputs_F(nn_t x);
extern nn_t nn_sort_outputs_F(nn_t x);


extern nn_t nn_append_singleton_dim_in_if_exists_F(nn_t op, const char* iname);
extern nn_t nn_append_singleton_dim_out_if_exists_F(nn_t op, const char* oname);

extern nn_t nn_stack_inputs_if_exists_F(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim);
extern nn_t nn_stack_outputs_if_exists_F(nn_t x, int a, const char* aname, int b, const char* bname, int stack_dim);

extern nn_t nn_mark_dup_if_exists_F(nn_t x, const char* name);
extern nn_t nn_mark_stack_input_if_exists_F(nn_t x, const char* name);
extern nn_t nn_mark_stack_output_if_exists_F(nn_t x, const char* name);

extern nn_t nn_real_input(nn_t op, int i, const char* iname);
extern nn_t nn_real_output(nn_t op, int o, const char* oname);
extern nn_t nn_real_input_F(nn_t op, int i, const char* iname);
extern nn_t nn_real_output_F(nn_t op, int o, const char* oname);

#endif