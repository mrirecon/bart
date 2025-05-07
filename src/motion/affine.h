#include "misc/debug.h"

struct nlop_s;

const struct nlop_s* nlop_affine_chain_FF(const struct nlop_s* A, const struct nlop_s* B);
const struct nlop_s* nlop_affine_prepend_FF(const struct nlop_s* A, _Complex float* B);
const struct nlop_s* nlop_affine_append_FF(_Complex float* A, const struct nlop_s* B);
const struct nlop_s* nlop_affine_to_grid_F(const struct nlop_s* affine, const long sdims[3], const long mdims[3]);

void affine_init_id(_Complex float* dst);

const struct nlop_s* nlop_affine_translation_2D(void);
const struct nlop_s* nlop_affine_translation_3D(void);

const struct nlop_s* nlop_affine_rotation_2D(void);
const struct nlop_s* nlop_affine_rotation_3D(void);

const struct nlop_s* nlop_affine_rigid_2D(void);
const struct nlop_s* nlop_affine_rigid_3D(void);

const struct nlop_s* nlop_affine_2D(void);
const struct nlop_s* nlop_affine_3D(void);

extern void affine_debug(enum debug_levels dl, const _Complex float* A);

extern void affine_interpolate(int ord, const _Complex float* affine, const long _odims[3], _Complex float* dst, const long _idims[3], const _Complex float* src);

extern const struct nlop_s* nlop_affine_compute_pos(int dim, int N, const long sdims[N], const long mdims[N], const struct nlop_s* affine);

extern void affine_reg(_Bool gpu, _Bool cubic, _Complex float* affine, const struct nlop_s* trafo, long sdims[3], const _Complex float* img_static, const _Complex float* msk_static, long mdims[3], const _Complex float* img_moving, const _Complex float* msk_moving,
			int N, float sigma[N], float factor[N]);


