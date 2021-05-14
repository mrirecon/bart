/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#ifndef VDPRINTF_WINDOWS
#define VDPRINTF_WINDOWS

#include <stdio.h>

#include "win/vdprintf.h"
#include "misc/misc.h"

#include <stdio.h>

int vdprintf(int fd, const char *format, va_list ap);

int vdprintf(int fd, const char *format, va_list ap)
{
	FILE* stream = _fdopen(fd, "a");
	if (stream == NULL)
		error("Unable to open file.\n");
	int err = vfprintf(stream, format, ap);
	if (err == -1)
		error("Unable to write to file.\n");
	return err;
}

#endif /*  VDPRINTF_WINDOWS */
