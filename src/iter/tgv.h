
struct reg {

	const struct linop_s* linop;
	const struct operator_p_s* prox;
};

struct reg2 {

	const struct linop_s* linop[2];
	const struct operator_p_s* prox[2];
};

extern struct reg tv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long img_dims[N], int tvscales_N, const float tvscales[tvscales_N]);
extern struct reg2 tgv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift, const float alpha[2], int tvscales_N, const float tvscales[tvscales_N]);
extern struct reg2 ictv_reg(unsigned long flags1, unsigned long flags2, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift);



