#ifndef _SEQ_FLASH_H
#define _SEQ_FLASH_H

#include "misc/cppwrap.h"

#include "seq/event.h"


#define ERROR_LIST \
	X(ERROR_SETTING_DIM, -102)			\
	X(ERROR_SETTING_SPOKES_RAGA, -122)		\
	X(ERROR_SETTING_SPOKES_EVEN, -123)		\
	X(ERROR_PREP_GRAD_SLI, -201)			\
	X(ERROR_PREP_GRAD_SLI_REPH, -202)		\
	X(ERROR_SLI_TIMING, -221)			\
	X(ERROR_PREP_GRAD_RO_DEPH, -301)		\
	X(ERROR_PREP_GRAD_RO_RO, -302)			\
	X(ERROR_PREP_GRAD_RO_BLIP, -303)		\
	X(ERROR_BLIP_TIMING, -310)			\
	X(ERROR_RO_TIMING, -321)			\
	X(ERROR_ROT_ANGLE, -341)			\
	X(ERROR_MAX_GRAD_RO_SLI, -351)			\
	X(ERROR_END_FLAT_KERNEL, -901)

enum seq_error {
#define X(name, val) name = val,
    ERROR_LIST
#undef X
};

static inline const char *error_string(enum seq_error e) {
    switch (e) {
#define X(name, val) case name: return #name;
        ERROR_LIST
#undef X
        default: return "ERROR_UNKNOWN";
    }
}


struct seq_config;

extern double min_tr_flash(const struct seq_config* seq);
extern void min_te_flash(const struct seq_config* seq, double* min_te, double* fil_te);

extern int flash(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_FLASH_H
