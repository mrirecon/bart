#ifndef __SEQ_UI_ENUMS_H
#define __SEQ_UI_ENUMS_H

#include "seq/custom_ui.h"

enum custom_idx_long {

	// enum
	cil_pe_mode = _cil_pe_mode,
	cil_contrast = _cil_contrast,

	// bool
	cil_sms,

	// long
	cil_sms_distance = _cil_sms_distance,

	// long array
	cil_tiny = _cil_tiny,
	cil_rf_duration,
	cil_init_delay,
	cil_inversions,
	cil_inv_delay,
	cil_mb_factor,
	cil_RAGA_aligned_flags,

}; // max 64



enum custom_idx_double {

	// double

	// double Array
	cid_BWTP,
	cid_dummy = _cid_dummy

}; // max 16

#endif // __SEQ_UI_ENUMS_H
