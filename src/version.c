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
	"Print BART version string\n\n"
	"-h\thelp\n";
			

int main_version(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 0, usage_str, help_str);

	printf("%s\n", bart_version);

	exit(0);
}



