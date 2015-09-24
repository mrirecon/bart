/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2015 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <complex.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>

#include "num/multind.h"
#include "misc/misc.h"

#include "io.h"


int write_cfl_header(int fd, int n, const long dimensions[n])
{
	char header[4096];
	memset(header, 0, 4096);

	int pos = 0;

	pos += snprintf(header + pos, 4096 - pos, "# Dimensions\n");

	for (int i = 0; i < n; i++)
		pos += snprintf(header + pos, 4096 - pos, "%ld ", dimensions[i]);

	pos += snprintf(header + pos, 4096 - pos, "\n");

	if (pos != write(fd, header, pos))
		return -1;

	return 0;
}



int read_cfl_header(int fd, int n, long dimensions[n])
{
	char header[4097];
	memset(header, 0, 4097);

	int max;
	if (0 > (max = read(fd, header, 4096)))
		return -1;

	int pos = 0;
	int delta;
	bool ok = false;

	while (true) {

		// skip lines not starting with '#'

		while ('#' != header[pos]) {

			if ('\0' == header[pos])
				goto out;

			if (0 != sscanf(header + pos, "%*[^\n]\n%n", &delta))
				return -1;

			pos += delta;
		}

		char keyword[32];

		if (1 == sscanf(header + pos, "# %31s\n%n", keyword, &delta)) {

			pos += delta;

			if (0 == strcmp(keyword, "Dimensions")) {

				for (int i = 0; i < n; i++)
					dimensions[i] = 1;

				long val;
				int i = 0;

				while (1 == sscanf(header + pos, "%ld%n", &val, &delta)) {

					pos += delta;

					if (i < n)
						dimensions[i] = val;
					else
					if (1 != val)
						return -1;

					i++;
				}

				if (0 != sscanf(header + pos, "\n%n", &delta))
					return -1;

				pos += delta;

				if (ok)
					return -1;

				ok = true;
			}
		}
	}

out:
	return ok ? 0 : -1;
}




int write_coo(int fd, int n, const long dimensions[n])
{
	char header[4096];
	memset(header, 0, 4096);

	int pos = 0;

	pos += snprintf(header, 4096, "Type: float\nDimensions: %d\n", n);

	long start = 0;
	long stride = 1;

	for (int i = 0; i < n; i++) {

		long size = dimensions[i];

		pos += snprintf(header + pos, 4096 - pos, "[%ld\t%ld\t%ld\t%ld]\n", start, stride * size, size, stride);
		stride *= size;
	}

	if (4096 != write(fd, header, 4096))
		return -1;

	return 0;
}


int read_coo(int fd, int n, long dimensions[n])
{
	char header[4096];

	if (4096 != read(fd, header, 4096))
		return -1;

	int pos = 0;
	int delta;

	if (0 != sscanf(header + pos, "Type: float\n%n", &delta))
		return -1;

	pos += delta;

	int dim;
	
	if (1 != sscanf(header + pos, "Dimensions: %d\n%n", &dim, &delta))
		return -1;

	pos += delta;

//	if (n != dim)
//		return -1;
	
	for (int i = 0; i < n; i++)
		dimensions[i] = 1;

	for (int i = 0; i < dim; i++) {

		long val;
		
		if (1 != sscanf(header + pos, "[%*d %*d %ld %*d]\n%n", &val, &delta))
			return -1;

		pos += delta;
		
		if (i < n)
			dimensions[i] = val;
		else
		if (1 != val)	// fail if we have to many dimensions not equal 1
			return -1;
	}

	return 0;
}





