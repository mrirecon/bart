/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */


#ifndef __GRECON_H
#define __GRECON_H 1
 

#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"


struct sense_conf;
struct ecalib_conf;


extern void w1sense(struct sense_conf* conf, const struct ecalib_conf* calib, _Bool rplksp, const long dims[5], const long ostr[DIMS], complex float* image, const long sens_dims[DIMS], const complex float* sens_maps, const long pat1_dims[5], const complex float* weights, const long istr[DIMS], const complex float* kspace_data, _Bool usegpu);

extern void msense(struct sense_conf* conf, const long dims[DIMS], _Complex float* image, const _Complex float* sens_maps, const _Complex float* kspace_data, _Bool usegpu);
extern void msense2(struct sense_conf* conf, const struct ecalib_conf* calib, _Bool rplksp, const long dims[DIMS], const long ostr[DIMS],  _Complex float* image, const long sens_dims[DIMS], const _Complex float* sens_maps, const long istr[DIMS], const _Complex float* kspace_data, _Bool usegpu);
extern void msense3(struct sense_conf* conf, const struct ecalib_conf* calib, const long dims[DIMS], _Complex float* image, const long sens_dims[DIMS], const _Complex float* sens_maps, const _Complex float* kspace_data, _Bool usegpu);
extern void msense2pocs(struct sense_conf* conf, const struct ecalib_conf* calib, const long dims[DIMS], _Complex float* image, const long sens_dims[DIMS], const _Complex float* sens_maps, const _Complex float* kspace_data, _Bool usegpu);

extern void mnoir(struct sense_conf* conf, const long dims[DIMS], _Complex float* image, const _Complex float* kspace_data, _Bool usegpu);



#ifdef __cplusplus
}
#endif


#endif // __GRECON_H

