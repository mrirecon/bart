/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2021-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/opts.h"


static const char help_str[] = "Convert between a bitmask and set of dimensions.";




int main_bitmask(int argc, char* argv[argc])
{
	int count = 0;
	unsigned long* dims = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(false, &count, 1, TUPLE_ULONG(&dims, "dim")),
	};

	bool inverse = false;

	const struct opt_s opts[] = {

		OPT_SET('b', &inverse, "dimensions from bitmask, use with exactly one argument"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if ((1 != count) && inverse)
		error("exactly one argument needed.\n");

	unsigned long flags = 0;

	if (!inverse) {

		for (int i = 0; i < count; i++) {

			int d = dims[i];

			flags = MD_SET(flags, d);
		}

		bart_printf("%ld\n", flags);

	} else {

		int i = 0;
		flags = dims[0];

		while (flags) {

			if (flags & 1)
				bart_printf("%d ", i);

			flags >>= 1;
			i++;
		}

		bart_printf("\n");
	}

	xfree(dims);

	return 0;
}


