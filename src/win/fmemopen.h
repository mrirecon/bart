/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#ifndef FMEMOPEN_WINDOWS
#define FMEMOPEN_WINDOWS

FILE *fmemopen(void *buf, size_t len, const char *type);

#endif
