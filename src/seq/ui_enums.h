#ifndef __SEQ_UI_ENUMS_H
#define __SEQ_UI_ENUMS_H

#include "seq/custom_ui.h"

#define SEQ_CUSTOM_UI_IDX_LONG(M)	\
	M(pe_mode)			\
	M(contrast)			\
	/* bool */			\
	M(reco)				\
	M(sms)				\
	/* long */			\
	/* long array */		\
	M(tiny)				\
	M(rf_duration)			\
	M(init_delay)			\
	M(inversions)			\
	M(inv_delay)			\
	M(mb_factor)			\
	M(RAGA_aligned_flags)

enum custom_idx_long {
#define enum_entry(name) cil_##name,
SEQ_CUSTOM_UI_IDX_LONG(enum_entry)
#undef enum_entry
}; // max 64


#define SEQ_CUSTOM_UI_IDX_DOUBLE(M)	\
	M(cmd)				\
	M(BWTP)				\

enum custom_idx_double {
#define enum_entry(name) cid_##name,
SEQ_CUSTOM_UI_IDX_DOUBLE(enum_entry)
#undef enum_entry
}; // max 16

#endif // __SEQ_UI_ENUMS_H
