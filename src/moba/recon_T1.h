
#ifndef RECON_T1_H
#define RECON_T1_H

#include "misc/mri.h"

struct noir_conf_s;
extern void T1_recon(const struct noir_conf_s* conf, const long dims[DIMS], _Complex float* img, _Complex float* sens, const _Complex float* pattern, const _Complex float* mask, const _Complex float* TI, const _Complex float* kspace_data, _Bool usegpu);


#endif
