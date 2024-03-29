struct nlop_s;

extern struct nlop_s* nlop_chain(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_chain_FF(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_chain2(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_chain2_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_chain2_swap_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_chain2_keep(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_chain2_keep_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_chain2_keep_swap_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_append_FF(const struct nlop_s* a, int o, const struct nlop_s* b);
extern struct nlop_s* nlop_prepend_FF(const struct nlop_s* a, const struct nlop_s* b, int i);
extern struct nlop_s* nlop_combine(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_combine_FF(const struct nlop_s* a, const struct nlop_s* b);
extern struct nlop_s* nlop_link(const struct nlop_s* x, int oo, int ii);
extern struct nlop_s* nlop_link_F(const struct nlop_s* x, int oo, int ii);
extern struct nlop_s* nlop_permute_inputs(const struct nlop_s* x, int I2, const int perm[__VLA(I2)]);
extern struct nlop_s* nlop_permute_outputs(const struct nlop_s* x, int O2, const int perm[__VLA(O2)]);
extern struct nlop_s* nlop_permute_inputs_F(const struct nlop_s* x, int I2, const int perm[__VLA(I2)]);
extern struct nlop_s* nlop_permute_outputs_F(const struct nlop_s* x, int O2, const int perm[__VLA(O2)]);
extern struct nlop_s* nlop_dup(const struct nlop_s* x, int a, int b);
extern struct nlop_s* nlop_dup_F(const struct nlop_s* x, int a, int b);
extern struct nlop_s* nlop_stack_inputs(const struct nlop_s* x, int a, int b, int stack_dim);
extern struct nlop_s* nlop_stack_inputs_F(const struct nlop_s* x, int a, int b, int stack_dim);
extern struct nlop_s* nlop_stack_outputs(const struct nlop_s* x, int a, int b, int stack_dim);
extern struct nlop_s* nlop_stack_outputs_F(const struct nlop_s* x, int a, int b, int stack_dim);
extern struct nlop_s* nlop_shift_input(const struct nlop_s* x, int new_index, int old_index);
extern struct nlop_s* nlop_shift_input_F(const struct nlop_s* x, int new_index, int old_index);
extern struct nlop_s* nlop_shift_output(const struct nlop_s* x, int new_index, int old_index);
extern struct nlop_s* nlop_shift_output_F(const struct nlop_s* x, int new_index, int old_index);

extern struct nlop_s* nlop_stack_multiple_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO], _Bool container, _Bool multigpu);
extern struct nlop_s* nlop_stack_inputs_generic_F(const struct nlop_s* x, int NI, int _index[__VLA(NI)], int stack_dim);
extern struct nlop_s* nlop_stack_outputs_generic_F(const struct nlop_s* x, int NO, int _index[__VLA(NO)], int stack_dim);

