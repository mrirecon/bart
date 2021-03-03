#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "noncart/nufft.h"

#include "linops/linop.h"


#include "misc.h"


struct network_data_s network_data_empty = {

	.kdims = { 0 },
	.cdims = { 0 },
	.pdims = { 0 },
	.idims = { 0 },

	.filename_trajectory = NULL,
	.filename_pattern = NULL,
	.filename_kspace = NULL,
	.filename_coil = NULL,
	.filename_out = NULL,

	.kspace = NULL,
	.coil = NULL,
	.pattern = NULL,
	.out = NULL,

	.create_out = false,
	.load_mem = false,
};


void load_network_data(struct network_data_s* nd) {


	nd->coil = load_cfl(nd->filename_coil, DIMS, nd->cdims);

	if (nd->load_mem) {

		complex float* tmp = anon_cfl("", DIMS, nd->cdims);
		md_copy(DIMS, nd->cdims, tmp, nd->coil, CFL_SIZE);
		unmap_cfl(DIMS, nd->cdims, nd->coil);
		nd->coil = tmp;
	}

	md_copy_dims(DIMS, nd->kdims, nd->cdims);
	md_select_dims(DIMS, ~COIL_FLAG, nd->idims, nd->cdims);

	if (nd->create_out) {

		nd->out = create_cfl(nd->filename_out, DIMS, nd->idims);
	} else {

		long idims_file[DIMS];
		nd->out = load_cfl(nd->filename_out, DIMS, idims_file);
		assert(md_check_equal_dims(DIMS, nd->idims, idims_file, ~0));

		if (nd->load_mem) {

			complex float* out_tmp = anon_cfl("", DIMS, nd->idims);
			md_copy(DIMS, nd->idims, out_tmp, nd->out, CFL_SIZE);
			unmap_cfl(DIMS, nd->idims, nd->out);
			nd->out = out_tmp;
		}
	}

	long kdims_file[DIMS];
	complex float* kspace_file = load_cfl(nd->filename_kspace, DIMS, kdims_file);

	complex float* pattern_file = NULL;
	long pdims_file[DIMS];

	if (NULL != nd->filename_pattern) {

		pattern_file = load_cfl(nd->filename_pattern, DIMS, pdims_file);
	} else {

		md_select_dims(DIMS, ~(COIL_FLAG), pdims_file, kdims_file);
		pattern_file = anon_cfl("", DIMS, pdims_file);
		estimate_pattern(DIMS, kdims_file, COIL_FLAG, pattern_file, kspace_file);
	}

	complex float* traj = NULL;
	long trj_dims[DIMS];


	if (NULL != nd->filename_trajectory) {

		traj = load_cfl(nd->filename_trajectory, DIMS, trj_dims);
		md_zsmul(DIMS, trj_dims, traj, traj, 2.);

		for (unsigned int i = 0; i < DIMS; i++)
			if (MD_IS_SET(FFT_FLAGS, i) && (1 < nd->kdims[i]))
				nd->kdims[i] *= 2;

		md_copy_dims(DIMS - 3, nd->kdims + 3, kdims_file + 3);
		md_copy_dims(3, nd->pdims, nd->kdims);
		md_copy_dims(DIMS - 3, nd->pdims + 3, trj_dims + 3);

		unsigned int cmp_flag = md_nontriv_dims(DIMS, pdims_file) & md_nontriv_dims(DIMS, kdims_file);

		if (!md_check_equal_dims(DIMS, pdims_file, kdims_file, cmp_flag)) {

			debug_printf(DP_WARN, "pdims: ");
			debug_print_dims(DP_INFO, DIMS, pdims_file);
			debug_printf(DP_WARN, "kdims: ");
			debug_print_dims(DP_INFO, DIMS, kdims_file);
			error("Inconsistent dimensions of kspace and pattern!");
		}

		//compute psf

		nd->pattern = compute_psf(DIMS, nd->pdims, trj_dims, traj, trj_dims, NULL, pdims_file, pattern_file, false, false);
		fftuc(DIMS, nd->pdims, FFT_FLAGS, nd->pattern, nd->pattern);

		float pattern_scale = 1.;
		float kspace_scale = 1.;

		for (int i = 0; i < 3; i++)
			if (1 != nd->pdims[i]) {

				pattern_scale *= 2.;
				kspace_scale *= sqrtf(2.);
			}

		md_zsmul(DIMS, nd->pdims, nd->pattern, nd->pattern, pattern_scale);

		//grid kspace

		nd->kspace = anon_cfl("", DIMS, nd->kdims);

		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = false;
		nufft_conf.lowmem = true;

		auto nufft_op = nufft_create(DIMS, kdims_file, nd->kdims, trj_dims, traj, NULL, nufft_conf);
		linop_adjoint(nufft_op, DIMS, nd->kdims, nd->kspace, DIMS, kdims_file, kspace_file);
		linop_free(nufft_op);

		fftuc(DIMS, nd->kdims, FFT_FLAGS, nd->kspace, nd->kspace);
		md_zsmul(DIMS, nd->kdims, nd->kspace, nd->kspace, kspace_scale);

		unmap_cfl(DIMS, kdims_file, kspace_file);

	} else {

		md_copy_dims(DIMS, nd->kdims, kdims_file);

		if (nd->load_mem) {

			nd->kspace = anon_cfl("", DIMS, nd->kdims);
			md_copy(DIMS, nd->kdims, nd->kspace, kspace_file, CFL_SIZE);
			unmap_cfl(DIMS, kdims_file, kspace_file);
		} else {

			nd->kspace = kspace_file;
		}

		md_copy_dims(DIMS, nd->pdims, pdims_file);
		nd->pattern = md_alloc(DIMS, nd->pdims, CFL_SIZE);
		md_copy(DIMS, nd->pdims, nd->pattern, pattern_file, CFL_SIZE);
	}
}

void free_network_data(struct network_data_s* nd)
{
	md_free(nd->pattern);
	unmap_cfl(DIMS, nd->kdims, nd->kspace);
	unmap_cfl(DIMS, nd->cdims, nd->coil);
	unmap_cfl(DIMS, nd->idims, nd->out);
}

void network_data_check_simple_dims(struct network_data_s* network_data)
{
	bool consistent_dims = true;

	for (unsigned int i = 3; i < DIMS; i++)
		consistent_dims = consistent_dims && (network_data->kdims[i] == network_data->cdims[i]);
	for (int i = 0; i < 3; i++)
		consistent_dims = consistent_dims && (network_data->kdims[i] == network_data->pdims[i]);

	consistent_dims = consistent_dims && (1 == network_data->pdims[3]);
	consistent_dims = consistent_dims && ((1 == network_data->pdims[4]) || (network_data->kdims[4] == network_data->pdims[4]));

	if (!consistent_dims) {

		debug_printf(DP_WARN, "kdims: ");
		debug_print_dims(DP_INFO, DIMS, network_data->kdims);
		debug_printf(DP_WARN, "cdims: ");
		debug_print_dims(DP_INFO, DIMS, network_data->cdims);
		debug_printf(DP_WARN, "pdims: ");
		debug_print_dims(DP_INFO, DIMS, network_data->pdims);
		debug_printf(DP_WARN, "idims: ");
		debug_print_dims(DP_INFO, DIMS, network_data->idims);

		error("Inconsistent dimensions!");
	}
}