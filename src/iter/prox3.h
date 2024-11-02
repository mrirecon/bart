
#ifndef __PROX3_H
#define __PROX3_H

#include "misc/cppwrap.h"

struct operator_p_s;
extern const struct operator_p_s* prox_convex_conjugate(const struct operator_p_s* prox);
extern const struct operator_p_s* prox_convex_conjugate_F(const struct operator_p_s* prox);

#include "misc/cppwrap.h"

#endif

