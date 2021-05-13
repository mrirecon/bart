/* Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#ifndef VDPRINTF_WINDOWS
#define VDPRINTF_WINDOWS

#include <stdio.h>

int vdprintf(int fd, const char *format, va_list ap);

#endif /*  VDPRINTF_WINDOWS */
