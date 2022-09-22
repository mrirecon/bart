/* Copyright 2021. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "version.h"

#define STRINGIFY(x) # x 
#define VERSION(x) STRINGIFY(x)
const char* bart_version = 
#include "version.inc"
;


bool version_parse(unsigned int v[5], const char* version)
{
	int q, r, s;
	s = -1; //initialize variabel;

	int len = strlen(version);

	v[3] = 0;	// patch level

	// simple version string format, for when git describe fails
	// This might happen if the .git directory exsits, but git is not installed on a system
	int ret = sscanf(version, "v%u.%u.%u-dirty%n", &v[0], &v[1], &v[2], &s);

	if ((3 == ret) && (len == s)) {

		v[4] = 1; 	// dirty
		return true;
	}


	ret = sscanf(version, "v%u.%u.%u%n-%u-g%*40[0-9a-f]%n-dirty%n", &v[0], &v[1], &v[2], &q, &v[3], &r, &s);

	if (!(   ((3 == ret) && (len == q))
		|| ((4 == ret) && (len == r))
		|| ((4 == ret) && (len == s))))
		return false;

	for (int i = 0; i < 4; i++)
		if (v[i] >= 1000000)
			return false;

	v[4] = (len == s) ? 1 : 0;	// dirty

	return true;
}


int version_compare(const unsigned int va[5], const unsigned int vb[5])
{
	for (int i = 0; i < 5; i++) { // lexicographical comparison

		if (va[i] > vb[i])
			return 1;

		if (va[i] < vb[i])
			return -1;
	}

	return 0;
}



unsigned int requested_compat_version[5] = { UINT_MAX, 0, 0, 0, 0 };

bool use_compat_to_version(const char* check_version)
{

	if (UINT_MAX == requested_compat_version[0]) {

		char* str = getenv("BART_COMPAT_VERSION");

		if (NULL == str)
			return false;

		if (!version_parse(requested_compat_version, str)) {

			debug_printf(DP_WARN, "Could not parse BART_COMPAT_VERSION, ignoring it!\n");
			return false;
		}

		debug_printf(DP_INFO, "Setting compatibility version to: %s\n", str);
	}


	unsigned int check_compat_version[5];

	if (!version_parse(check_compat_version, check_version))
		assert(0);


	return (version_compare(check_compat_version, requested_compat_version) >= 0);
}

