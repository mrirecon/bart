#include "misc/mri.h"

struct nlop_s;
struct noir_model_conf_s;

enum fat_spec;

#ifndef _MECO_MODEL
#define _MECO_MODEL 1
enum meco_model {
	MECO_WF,
	MECO_WFR2S,
	MECO_WF2R2S,
	MECO_R2S,
	MECO_PHASEDIFF,
	MECO_PI,
	IR_MECO_WF_fB0,
	IR_MECO_WF_R2S,
	IR_MECO_T1_R2S,
	IR_MECO_W_T1_F_T1_R2S,
};
#endif


void ir_meco_calc_fat_modu(int N, const long dims[N], const complex float TE[dims[CSHIFT_DIM]], complex float dst[dims[CSHIFT_DIM]], enum fat_spec fat_spec);

extern const struct linop_s* ir_meco_get_fB0_trafo(struct nlop_s* op);
extern void ir_meco_forw_fB0(const struct linop_s* op, complex float* dst, const complex float* src);
extern void ir_meco_back_fB0(const struct linop_s* op, complex float* dst, const complex float* src);

extern int ir_meco_get_num_of_coeff(enum meco_model sel_model);

extern struct nlop_s* nlop_ir_meco_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N],
		const complex float* TI, const long TE_dims[N], const complex float* TE, const float* scale_fB0, const float* scale);
