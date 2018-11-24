/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017	Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>

#include "num/filter.h"
#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/misc.h"

#include "utest.h"


static bool test_window(unsigned int D, long dims[D], long flags, bool hamming, const complex float* ref)
{

	complex float* in = md_alloc(3, dims, CFL_SIZE);
	md_zfill(3, dims, in, 1.);

	complex float* out = md_alloc(3, dims, CFL_SIZE);

	(hamming ? md_zhamming : md_zhann)(3, dims, flags, out, in);

	bool ret = (md_znrmse(3, dims, &ref[0], out) < UT_TOL);

	md_free(out);
	md_free(in);
	
	return ret;
}

static bool test_hamming(void)
{
	// compare to Matlab:
	// >> z = permute(repmat(hamming(5)*hamming(4)', [1, 1, 2]), [3, 1, 2]);
	// >> z(:)
	const complex float ref[2 * 5 * 4] = {

		 0.0064, 0.0064, 0.0432, 0.0432,
		 0.0800, 0.0800, 0.0432, 0.0432,
		 0.0064, 0.0064, 0.0616, 0.0616,
		 0.4158, 0.4158, 0.7700, 0.7700,
		 0.4158, 0.4158, 0.0616, 0.0616,
		 0.0616, 0.0616, 0.4158, 0.4158,
		 0.7700, 0.7700, 0.4158, 0.4158,
		 0.0616, 0.0616, 0.0064, 0.0064,
		 0.0432, 0.0432, 0.0800, 0.0800,
		 0.0432, 0.0432, 0.0064, 0.0064,
	};

	long dims[3] = { 2, 5, 4 };

	return test_window(3, dims, MD_BIT(1) | MD_BIT(2), true, ref);
}

UT_REGISTER_TEST(test_hamming);

static bool test_hann(void)
{

	// compare to Matlab:
	// >> z = permute(repmat(hann(4)*hann(5)', [1, 1, 2]), [1, 3, 2]);
	// >> z(:)
	const complex float ref[4 * 2 * 5] = {

		0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.3750, 0.3750, 0.0000,
		0.0000, 0.3750, 0.3750, 0.0000,
		0.0000, 0.7500, 0.7500, 0.0000,
		0.0000, 0.7500, 0.7500, 0.0000,
		0.0000, 0.3750, 0.3750, 0.0000,
		0.0000, 0.3750, 0.3750, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000,
	};

	long dims[3] = { 4, 2, 5 };

	return test_window(3, dims, MD_BIT(0) | MD_BIT(2), false, ref);
}

UT_REGISTER_TEST(test_hann);

