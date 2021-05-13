/* Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#include <stdio.h>

#include "win/vdprintf.h"
#include "misc/misc.h"

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
