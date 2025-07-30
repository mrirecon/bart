
#ifndef _SEQ_HELPERS_H
#define _SEQ_HELPERS_H

#include "misc/cppwrap.h"

struct seq_config;

extern void set_loop_dims_and_sms(struct seq_config* seq, long partitions, long total_slices, long radial_views,
	long frames, long echoes, long inv_reps, long phy_phases, long averages, int sms, long mb_factor);

#include "misc/cppwrap.h"

#endif // _SEQ_HELPERS_H
