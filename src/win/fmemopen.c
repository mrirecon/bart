/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#include <windows.h>
#include <stdio.h>
#include "misc/misc.h"
#include "win/fmemopen.h"

FILE *fmemopen(void *buf, size_t len, const char *type)
{
    TCHAR temp_path[MAX_PATH];
    TCHAR temp_file_name[MAX_PATH];
	FILE *stream;
	
	DWORD dwRetVal = GetTempPath(MAX_PATH, temp_path);
	if (dwRetVal > MAX_PATH || (dwRetVal == 0))
        error("Failed to get temp dir");
    if (!GetTempFileName(temp_path, TEXT("bart_"), 0, temp_file_name))
		error("Failed to get temporary file name");

	char ext_type[strlen(type) + 2];
	strcat(ext_type, type);
	// "T": Specifies a file as temporary. If possible, it isn't flushed to disk.
	// "D": Specifies a file as temporary. It's deleted when the last file pointer is closed.
	strcat(ext_type, "TD");

	if (fopen_s( &stream, temp_file_name, ext_type))
        error("Failed to open temp file as output buffer");

	return stream;
}
