#ifndef _ITER_TGV_H
#define _ITER_TGV_H

#include "linops/linop.h"
struct reg {

	const struct linop_s* linop;
	const struct operator_p_s* prox;
};

struct reg2 {

	const struct linop_s* linop[2];
	const struct operator_p_s* prox[2];
};

struct reg4 {

	const struct linop_s* linop[4];
	const struct operator_p_s* prox[4];
};

extern struct reg tv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long img_dims[N], int tvscales_N, const float tvscales[tvscales_N], const struct linop_s* lop_trafo);
extern struct reg2 tgv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift, const float alpha[2], int tvscales_N, const float tvscales[tvscales_N], const struct linop_s* lop_trafo);
extern struct reg2 ictv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift, const float gamma[2], int tvscales_N, const float tvscales[tvscales_N], int tvscales2_N, const float tvscales2[tvscales2_N], const struct linop_s* lop_trafo);
extern struct reg4 ictgv_reg(unsigned long flags, unsigned long jflags, float lambda, int N, const long in_dims[N], long isize, long* ext_shift, const float alpha[2], const float gamma[2], int tvscales_N, const float tvscales[tvscales_N], int tvscales2_N, const float tvscales2[tvscales2_N], const struct linop_s* lop_trafo);

#endif // _ITER_TGV_H

