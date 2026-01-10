/* Copyright 2023-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <string.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/iovec.h"

#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/io.h"

#include "nlops/nlop.h"

#include "nn/tf_wrapper.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] =
	"Load Tensorflow Graph";



int main_tensorflow(int argc, char* argv[argc])
{
	int count = 0;
	const char* graph = NULL;
	const char* key = NULL;

	const char** files = NULL;
	long batchsize = 1;
	bool nodes = false;
	
	struct arg_s args[] = {

		ARG_STRING(true, &graph, "TensorFlow Graph"),
		ARG_TUPLE(false, &count, 1, { OPT_INOUTFILE, sizeof(char*), &files, "Arguments" })
	};

	const struct opt_s opts[] = {
		OPT_LONG('b', &batchsize, "b", "Fill placeholder in dims with b"),
		OPT_SET('n', &nodes, "Print all nodes in graph"),
		OPT_SET('g', &bart_use_gpu, "Use gpu"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init_gpu_support();

	const struct tf_shared_graph_s* sgraph = tf_shared_graph_create(graph, key);
	tf_shared_graph_set_batch_size(sgraph, batchsize);

	if (nodes)
		tf_shared_graph_list_operations(sgraph);

	const struct nlop_s* nlop = nlop_tf_shared_create(sgraph);
	tf_shared_graph_free(sgraph);

	if (0 == count) {

		nlop_debug(DP_INFO, nlop);

	} else {

		int II = nlop_get_nr_in_args(nlop);
		int OO = nlop_get_nr_out_args(nlop);

		assert(count == II + OO);

		long dims[count][DIMS];
		
		void* args[count];

		for (int i = 0; i < II; i++) {

			args[OO + i] = load_cfl(files[i], DIMS, dims[OO + i]);
			auto dom = nlop_generic_domain(nlop, i);
			assert(DIMS >= dom->N);
			assert(md_check_equal_dims(dom->N, dom->dims, dims[OO + i], ~0UL));
		}

		for (int i = 0; i < OO; i++) {

			auto dom = nlop_generic_codomain(nlop, i);
			assert(DIMS >= dom->N);
			md_singleton_dims(DIMS, dims[i]);
			md_copy_dims(dom->N, dims[i], dom->dims);

			args[i] = create_cfl(files[II + i], dom->N, dims[i]);			
		}

		nlop_generic_apply_unchecked(nlop, OO + II, args);

		for (int i = 0; i < OO + II; i++)
			unmap_cfl(DIMS, dims[i], args[i]);
	}

	nlop_free(nlop);
	xfree(files);

	return 0;
}

