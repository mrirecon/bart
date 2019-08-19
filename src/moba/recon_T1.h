
#ifndef RECON_T1_H
#define RECON_T1_H

#include "misc/mri.h"

struct moba_conf {

	unsigned int iter;
	float alpha;
	float alpha_min;
	float redu;
	float step;
	float lower_bound;
	float tolerance;
	unsigned int inner_iter;
	bool noncartesian;
};



extern struct moba_conf moba_defaults;


extern void T1_recon(const struct moba_conf* conf, const long dims[DIMS], _Complex float* img, _Complex float* sens, const _Complex float* pattern, const _Complex float* mask, const _Complex float* TI, const _Complex float* kspace_data, _Bool usegpu);


#endif
