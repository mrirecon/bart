
#ifndef _SEQ_ANGLE_CALC_H
#define _SEQ_ANGLE_CALC_H

#include "misc/cppwrap.h"

#ifndef DIMS
#define DIMS 16
#endif

struct seq_config;
struct traj_conf;

extern double get_rot_angle(const long pos[DIMS], const struct seq_config* seq);
extern void traj_conf_from_seq(struct traj_conf *conf, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_ANGLE_CALC_H
