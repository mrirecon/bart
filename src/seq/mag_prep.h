#ifndef _SEQ_MAG_PREP_H
#define _SEQ_MAG_PREP_H

#include "misc/cppwrap.h"

#include "seq/config.h"
#include "seq/event.h"

struct seq_config;
struct seq_state;

extern int mag_prep(struct seq_event ev[6], const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_MAG_PREP_H
