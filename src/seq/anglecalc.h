
#ifndef __ANGLE_CALC_H
#define __ANGLE_CALC_H

#include "misc/cppwrap.h"

#include "seq/config.h"


#ifndef DIMS
#define DIMS 16
#endif

double get_rot_angle(const long pos[DIMS], const struct seq_config* seq);


struct traj_conf;

void traj_conf_from_seq(struct traj_conf *conf, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // __ANGLE_CALC_H
