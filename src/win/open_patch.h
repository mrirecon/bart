/* Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#ifndef OPEN_PATCH_H
#define OPEN_PATCH_H
#include <share.h>
#define open(pathname, flags, ...) _sopen(pathname, flags|_O_BINARY, _SH_DENYNO __VA_OPT__(,) __VA_ARGS__)
#endif /* OPEN_PATCH_H */
