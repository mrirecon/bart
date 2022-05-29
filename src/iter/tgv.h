
struct reg2 {

	const struct linop_s* linop[2];
	const struct operator_p_s* prox[2];
};

extern struct reg2 tgv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], int* ext_shift);
extern struct reg2 ictv_reg(unsigned long flags1, unsigned long flags2, unsigned long jflags, float lambda, int N, const long in_dims[N], int* ext_shift);



