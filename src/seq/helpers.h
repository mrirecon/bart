
#ifndef _SEQ_HELPERS_H
#define _SEQ_HELPERS_H

#include "misc/dllspec.h"
#include "misc/cppwrap.h"

#include "misc/mri.h"

#define MAX_PARAMS_LONG 64
#define MAX_PARAMS_DOUBLE 16

// intended to include in sequence
// (besides config.h, event.h seq.h)
// we do not allow _Complex and bool here

// for interactive mode
BARTLIB_API int BARTLIB_CALL
seq_check_equal_dims(int D, const long dims1[__VLA(D)], const long dims2[__VLA(D)], unsigned long flags);

struct seq_config;

BARTLIB_API extern int BARTLIB_CALL seq_raga_spokes(const struct seq_config* seq);

// minimum TE and TR calculation at end of prepare
BARTLIB_API extern long BARTLIB_CALL seq_minimum_tr(const struct seq_config* seq);
BARTLIB_API extern void BARTLIB_CALL seq_minimum_te(const struct seq_config* seq, long* min_te, long* fil_te);


BARTLIB_API extern long BARTLIB_CALL seq_relevant_readouts_meas_time(const struct seq_config* seq);
BARTLIB_API extern double BARTLIB_CALL seq_total_measure_time(const struct seq_config* seq);


BARTLIB_API extern void BARTLIB_CALL set_loop_dims_and_sms(struct seq_config* seq, long partitions, long total_slices, long radial_views,
	long frames, long echoes, long phy_phases, long averages);

BARTLIB_API extern void BARTLIB_CALL
seq_ui_interface_custom_params(int reverse, struct seq_config* seq, int nl, long params_long[__VLA(nl)], int nd, double params_double[__VLA(nd)]);


BARTLIB_API extern long BARTLIB_CALL seq_get_slices(const struct seq_config* seq);
BARTLIB_API extern void BARTLIB_CALL seq_set_fov_pos(int N, int M, const float* shifts, struct seq_config* seq);



#include "misc/cppwrap.h"

#endif // _SEQ_HELPERS_H
