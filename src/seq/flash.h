#ifndef __SEQ_FLASH_H
#define __SEQ_FLASH_H

#include "misc/cppwrap.h"

#include "seq/config.h"
#include "seq/event.h"

enum seq_error { 

	ERROR_PREP_GRAD_SLI = -201,
	ERROR_PREP_GRAD_SLI_REPH = -202,
	ERROR_PREP_GRAD_RO_DEPH = -301,
	ERROR_PREP_GRAD_RO_RO = -302,
	ERROR_ROT_ANGLE = -341,
	ERROR_MAX_GRAD_RO_SLI = -351,
	ERROR_END_FLAT_KERNEL = -901,
};

void set_loop_dims_and_sms(struct seq_config* seq, long partitions, long total_slices, long radial_views,
	long frames, long echoes, long inv_reps, long phy_phases, long averages, int checkbox_sms, long mb_factor);

int flash(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // __SEQ_FLASH_H
