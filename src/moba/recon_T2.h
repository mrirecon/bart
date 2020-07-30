
#ifndef RECON_T2_H
#define RECON_T2_H

#include "misc/mri.h"
#include "recon_T1.h"

extern void T2_recon(const struct moba_conf* conf, const long dims[DIMS], _Complex float* img, _Complex float* sens, const _Complex float* pattern, const _Complex float* mask, const _Complex float* TI, const _Complex float* kspace_data, _Bool usegpu);

#endif
