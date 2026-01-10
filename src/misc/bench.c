/* Copyright 2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <string.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/gpuops.h"

#include "bench.h"


static void bench_sync(bool sync_gpu)
{
#ifdef USE_CUDA
	if (sync_gpu)
		cuda_sync_stream();
#else
	(void) sync_gpu;
#endif
	return;
}

void run_bench(long rounds, bool print, bool sync_gpu, bench_f fun)
{
	double runtimes[rounds];
	memset(runtimes, 0, sizeof runtimes); // maybe-uninitialized

	bench_sync(sync_gpu);

	for (long i = 0; i < rounds; ++i) {

		double tic = timestamp();

		NESTED_CALL(fun, ());
		bench_sync(sync_gpu);

		runtimes[i] = timestamp() - tic;
	}

	if (!print)
		return;

	double sum = 0.;
	double min = runtimes[0];
	double max = 0.;

	debug_printf(DP_DEBUG2, "Runtimes: ");

	for (long i = 0; i < rounds; ++i) {

		sum += runtimes[i];
		min = MIN(min, runtimes[i]);
		max = MAX(max, runtimes[i]);

		debug_printf(DP_DEBUG2, "%8.4f, ", runtimes[i]);
	}

	debug_printf(DP_DEBUG2, "\r\r\n");

	bart_printf("Avg: %8.4f Max: %8.4f Min: %8.4f\n", sum / (double)rounds, max, min);
}

