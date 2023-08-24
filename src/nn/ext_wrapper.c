/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */


#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "nlops/nlop.h"

#include "nn/pytorch_wrapper.h"
#include "nn/tf_wrapper.h"

#include "ext_wrapper.h"

const struct nlop_s* nlop_external_graph_create(const char* path, int OO, const int DO[OO], const long* odims[OO], int II, const int DI[II], const long* idims[II], bool init_gpu, const char* tf_signature_key)
{
	const struct nlop_s* nlop;

	if ((3 < strlen(path)) && (0 == strcmp(".pt", path + strlen(path) - 3))) {

		nlop = nlop_pytorch_create(path, II, DI, idims, init_gpu);
	} else {

		const struct tf_shared_graph_s* graph = tf_shared_graph_create(path, tf_signature_key);
		nlop = nlop_tf_shared_create(graph);

		assert(II == nlop_get_nr_in_args(nlop));

		long batch_size = 1;
		for (int i = 0; i < II; i++) {

			auto dom = nlop_generic_domain(nlop, i);

			if ((1 != batch_size) && (batch_size != md_calc_size(DI[i], idims[i]) / md_calc_size(dom->N, dom->dims)))
				error("Cannot find batch size to make TF-graph consistent with requested dimensions!\n");

			batch_size = md_calc_size(DI[i], idims[i]) / md_calc_size(dom->N, dom->dims);
		}

		if (1 != batch_size) {

			nlop_free(nlop);
			tf_shared_graph_set_batch_size(graph, batch_size);
			nlop = nlop_tf_shared_create(graph);
		}
	}

	for (int i = 0; i < II; i++) {

		auto dom = nlop_generic_domain(nlop, i);

		if (md_calc_size(DI[i], idims[i]) != md_calc_size(dom->N, dom->dims))
			error("Cannot find batch size to make graph consistent with requested dimensions!\n");

		bool shift_dim = bitcount(md_nontriv_dims(DI[i], idims[i])) == bitcount(md_nontriv_dims(dom->N, dom->dims));

		int j1 = 0;
		int j2 = 0;

		for (int j = 0; j < bitcount(md_nontriv_dims(DI[i], idims[i])); j++) {

			while (1 == idims[i][j1])
				j1++;

			while (1 == dom->dims[j2])
				j2++;

			shift_dim = shift_dim && idims[i][j1] == dom->dims[j2];
		}

		if (!shift_dim) {

			debug_printf(DP_WARN, "Cannot reshape graph's %dth input dims by simple shift!\n", i);
			debug_printf(DP_WARN, "Graph dims:    ");
			debug_print_dims(DP_INFO, dom->N, dom->dims);
			debug_printf(DP_WARN, "Requested dims:");
			debug_print_dims(DP_INFO, DI[i], idims[i]);
		}

		nlop = nlop_reshape_in_F(nlop, i, DI[i], idims[i]);
	}

	if (-1 != OO && OO != nlop_get_nr_out_args(nlop))
		error("Graph has %d outputs but %d are requested!\n", nlop_get_nr_out_args(nlop), OO);

	for (int i = 0; i < OO; i++) {

		auto dom = nlop_generic_domain(nlop, i);

		if (md_calc_size(DO[i], odims[i]) != md_calc_size(dom->N, dom->dims))
			error("Cannot find batch size to make graph consistent with requested dimensions!\n");

		bool shift_dim = bitcount(md_nontriv_dims(DO[i], odims[i])) == bitcount(md_nontriv_dims(dom->N, dom->dims));

		int j1 = 0;
		int j2 = 0;

		for (int j = 0; j < bitcount(md_nontriv_dims(DO[i], odims[i])); j++) {

			while (1 == odims[i][j1])
				j1++;

			while (1 == dom->dims[j2])
				j2++;

			shift_dim = shift_dim && odims[i][j1] == dom->dims[j2];
		}

		if (!shift_dim) {

			debug_printf(DP_WARN, "Cannot reshape graph's %dth output dims by simple shift!\n", i);
			debug_printf(DP_WARN, "Graph dims:    ");
			debug_print_dims(DP_INFO, dom->N, dom->dims);
			debug_printf(DP_WARN, "Requested dims:");
			debug_print_dims(DP_INFO, DO[i], odims[i]);
		}

		nlop = nlop_reshape_out_F(nlop, i, DO[i], odims[i]);
	}

	return nlop;
}

