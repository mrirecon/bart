/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#ifndef BASENAME_PATCH_H
#define BASENAME_PATCH_H

#include <stdlib.h>
#include <string.h>

char* win_basename(const char *path);

char* win_basename(const char *path) {
    char* substr = strrchr(path, '\\');
    if (NULL == substr)
        substr = strrchr(path, '/');
    return NULL == substr ? path : substr + 1;
}

#define basename(path) win_basename(path)

#endif /* BASENAME_PATCH_H */
