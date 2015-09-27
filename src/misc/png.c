/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <stdio.h>

#include <png.h>

#include "misc/misc.h"

#include "png.h"


int png_write_rgb24(const char* name, unsigned int w, unsigned int h, long inum, const unsigned char* buf)
{
	UNUSED(inum);

	FILE* fp;
	png_structp structp = NULL;
	png_infop infop = NULL;
	png_bytep* volatile row_ptrs = NULL;
	volatile int ret = -1;	// default: return failure

	if (NULL == (fp = fopen(name, "wb")))
		return -1;

	if (NULL == (structp = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL)))
		goto cleanup;

	if (NULL == (infop = png_create_info_struct(structp)))
		goto cleanup;

	if (setjmp(png_jmpbuf(structp)))
        	goto cleanup;

	png_set_IHDR(structp, infop, w, h, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_init_io(structp, fp);
	png_write_info(structp, infop);

	row_ptrs = xmalloc(sizeof(png_bytep) * h);
	int row_size = png_get_rowbytes(structp, infop);

	for (unsigned int i = 0; i < h; i++)
		row_ptrs[i] = (png_bytep)(buf + row_size * i);

	png_write_image(structp, row_ptrs);
	png_write_end(structp, infop);

	ret = 0;	// return success

cleanup:
	if (NULL != structp)
		png_destroy_write_struct(&structp, &infop);

	if (NULL != row_ptrs)
		free(row_ptrs);

	fclose(fp);
	return ret;
}

