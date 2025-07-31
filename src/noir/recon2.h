#include "misc/cppwrap.h"

#include "misc/mri.h"


struct noir2_conf_s {

	unsigned int iter;
	_Bool rvc;
	float alpha;
	float alpha_min;
	float redu;
	float a;
	float b;
	float c;

	float oversampling_coils;
	_Bool ret_os_coils;

	int phasepoles;

	_Bool sms;

	float scaling;
	_Bool undo_scaling;
	_Bool normalize_lowres;

	_Bool noncart;
	struct nufft_conf_s* nufft_conf;

	struct opt_reg_s* regs;

	_Bool gpu;

	int cgiter;
	float cgtol;

	unsigned long loop_flags;
	_Bool realtime;
	float temp_damp;

	_Bool legacy_early_stoppping;

	_Bool optimized;

	int iter_reg;
	int liniter;
	float lintol;
};

extern const struct noir2_conf_s noir2_defaults;

struct noir2_s;
extern void noir2_recon(const struct noir2_conf_s* conf, struct noir2_s* noir_ops,
			int N,
			const long img_dims[N], _Complex float* img, const _Complex float* img_ref,
			const long col_dims[N], _Complex float* sens,
			const long kco_dims[N], _Complex float* ksens, const _Complex float* sens_ref,
			const long ksp_dims[N], const _Complex float* kspace);

extern void noir2_recon_noncart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], _Complex float* img, const _Complex float* img_ref,
	const long col_dims[N], _Complex float* sens,
	const long kco_dims[N], _Complex float* ksens, const _Complex float* sens_ref,
	const long ksp_dims[N], const _Complex float* kspace,
	const long trj_dims[N], const _Complex float* traj,
	const long wgh_dims[N], const _Complex float* weights,
	const long bas_dims[N], const _Complex float* basis,
	const long msk_dims[N], const _Complex float* mask,
	const long cim_dims[N]);

extern void noir2_recon_cart(
	const struct noir2_conf_s* conf, int N,
	const long img_dims[N], _Complex float* img, const _Complex float* img_ref,
	const long col_dims[N], _Complex float* sens,
	const long kco_dims[N], _Complex float* ksens, const _Complex float* sens_ref,
	const long ksp_dims[N], const _Complex float* kspace,
	const long pat_dims[N], const _Complex float* pattern,
	const long bas_dims[N], const _Complex float* basis,
	const long msk_dims[N], const _Complex float* mask,
	const long cim_dims[N]);

#include "misc/cppwrap.h"

