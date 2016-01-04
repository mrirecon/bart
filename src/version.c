/* Copyright 2015. Martin Uecker 
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2015 Martin Uecker <uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/version.h"


static const char* usage_str = "[-h]";
static const char* help_str = 
	"Print BART version. The version string is of the form\n"
	"TAG or TAG-COMMITS-SHA as produced by 'git describe'. It\n"
	"specifies the last release (TAG), and (if git is used)\n"
	"the number of commits (COMMITS) since this release and\n"
	"the abbreviated hash of the last commit (SHA). If there\n"
	"are local changes '-dirty' is added at the end.\n";
			

int main_version(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 0, usage_str, help_str);

	printf("%s\n", bart_version);

	exit(0);
}



