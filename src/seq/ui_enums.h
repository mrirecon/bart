#ifndef __SEQ_UI_ENUMS_H
#define __SEQ_UI_ENUMS_H

#include "seq/custom_ui.h"

#define SEQ_CUSTOM_UI_IDX_LONG(M)	\
	M(PE_MODE)			\
	M(CONTRAST)			\
	/* bool */			\
	M(RECO)				\
	M(SMS)				\
	/* long */			\
	/* long array */		\
	M(TINY)				\
	M(RF_DURATION_US)		\
	M(INIT_DELAY)			\
	M(INVERSIONS)			\
	M(INV_DELAY)			\
	M(MB_FACTOR)			\
	M(RAGA_ALIGNED_FLAGS)

enum custom_idx_long {
#define enum_entry(name) SEQ_UI_IDX_LONG_##name,
SEQ_CUSTOM_UI_IDX_LONG(enum_entry)
#undef enum_entry
}; // max 64


#define SEQ_CUSTOM_UI_IDX_DOUBLE(M)	\
	M(CMD)				\
	M(BWTP)				\

enum custom_idx_double {
#define enum_entry(name) SEQ_UI_IDX_DOUBLE_##name,
SEQ_CUSTOM_UI_IDX_DOUBLE(enum_entry)
#undef enum_entry
}; // max 16

#endif // __SEQ_UI_ENUMS_H
