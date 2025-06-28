/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#ifndef _OPEN_PATCH_H
#define _OPEN_PATCH_H

#include <share.h>

#define open(pathname, flags, ...) _sopen(pathname, flags|_O_BINARY, _SH_DENYNO __VA_OPT__(,) __VA_ARGS__)

#endif /* _OPEN_PATCH_H */
