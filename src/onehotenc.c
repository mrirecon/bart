#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>
#include <math.h>

#include "misc/debug.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "nn/misc.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Transforms class labels to one-hot-encoded classes\n";


int main_onehotenc(int argc, char* argv[argc])
{
	bool reverse = false;
	int class_index = 0;

	const struct opt_s opts[] = {

		OPT_SET('r', &reverse, "get class label by maximum entry"),
		OPT_INT('i', &class_index, "index", "select dimension"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (!reverse) {

		long idims[DIMS];
		complex float* in = load_cfl(argv[1], DIMS, idims);

		while (1 != idims[class_index])
			class_index++;

		long odims[DIMS];
		md_copy_dims(DIMS, odims, idims);

		complex float max = in[0];
		md_zmax2(DIMS, idims, MD_SINGLETON_STRS(DIMS), &max, MD_SINGLETON_STRS(DIMS), &max, MD_STRIDES(DIMS, idims, CFL_SIZE), in);
		odims[class_index] = lroundf(max) + 1;

		complex float* out = create_cfl(argv[2], DIMS, odims);
		index_to_onehotenc(DIMS, odims, out, idims, in);

		unmap_cfl(DIMS, odims, out);
		unmap_cfl(DIMS, idims, in);

	} else {

		long idims[DIMS];
		complex float* in = load_cfl(argv[1], DIMS, idims);

		long odims[DIMS];
		md_select_dims(DIMS, ~MD_BIT(class_index), odims, idims);

		complex float* out = create_cfl(argv[2], DIMS, odims);
		onehotenc_to_index(DIMS, odims, out, idims, in);

		unmap_cfl(DIMS, odims, out);
		unmap_cfl(DIMS, idims, in);
	}

	return 0;
}


