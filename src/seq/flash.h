#ifndef _SEQ_FLASH_H
#define _SEQ_FLASH_H

#include "misc/cppwrap.h"

#include "seq/event.h"

enum seq_error { 

	ERROR_SETTING_DIM = -102,
	ERROR_SETTING_SPOKES_RAGA = -122,
	ERROR_SETTING_SPOKES_EVEN = -123,
	ERROR_PREP_GRAD_SLI = -201,
	ERROR_PREP_GRAD_SLI_REPH = -202,
	ERROR_SLI_TIMING = -221,
	ERROR_PREP_GRAD_RO_DEPH = -301,
	ERROR_PREP_GRAD_RO_RO = -302,
	ERROR_PREP_GRAD_RO_BLIP = -303,
	ERROR_BLIP_TIMING = -310,
	ERROR_RO_TIMING = -321,
	ERROR_ROT_ANGLE = -341,
	ERROR_MAX_GRAD_RO_SLI = -351,
	ERROR_END_FLAT_KERNEL = -901,
};

struct seq_config;

extern double min_tr_flash(const struct seq_config* seq);
extern void min_te_flash(const struct seq_config* seq, double* min_te, double* fil_te);

extern int flash(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_FLASH_H
