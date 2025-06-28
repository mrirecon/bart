
#include "misc/cppwrap.h"

#include "misc/mri.h"


struct noir_conf_s {

	unsigned int iter;
	_Bool rvc;
	_Bool noncart;
	float alpha;
	float alpha_min;
	float redu;
	float a;
	float b;
	_Bool pattern_for_each_coil;
	_Bool sms;
	unsigned int cnstcoil_flags;
	_Bool img_space_coils;
};

extern const struct noir_conf_s noir_defaults;

extern void noir_recon(const struct noir_conf_s* conf,
		       const long dims[DIMS],
		       _Complex float* img,
		       _Complex float* sens,
		       _Complex float* ksens,
		       const _Complex float* ref,
		       const _Complex float* pattern,
		       const _Complex float* mask,
		       const _Complex float* kspace_data);


#include "misc/cppwrap.h"

