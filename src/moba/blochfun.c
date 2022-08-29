/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <complex.h>
#include <math.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "noir/model.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "nlops/nlop.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "simu/bloch.h"

#include "moba/moba.h"

#include "noir/utils.h"

#include "blochfun.h"

#define round(x)	((int) ((x) + .5))



struct blochfun_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* der_dims;
	const long* map_dims;
	const long* in_dims;
	const long* out_dims;

	const long* der_strs;
	const long* map_strs;
	const long* in_strs;
	const long* out_strs;

	//derivatives
	complex float* derivatives;
	complex float* tmp;
	complex float* in_tmp;

	const complex float* b1;
	const complex float* slice;

	const struct moba_conf_s* moba_data;

	bool use_gpu;

	int counter;

	const complex float* weights;
	const struct linop_s* linop_alpha;
};

DEF_TYPEID(blochfun_s);



const struct linop_s* bloch_get_alpha_trafo(const struct nlop_s* op)
{
	struct blochfun_s* data = CAST_DOWN(blochfun_s, nlop_get_data(op));

	return data->linop_alpha;
}


void bloch_forw_alpha(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void bloch_back_alpha(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}


static void bloch_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	double starttime = timestamp();

	debug_printf(DP_DEBUG2, "Started Forward Calculation\n");

	struct blochfun_s* data = CAST_DOWN(blochfun_s, _data);

#if 0
	// Print out intermediate reconstruction steps

	if (DP_DEBUG2 <= debug_level) {

		// B1 map: kspace -> pixel

		long index[DIMS];
		md_set_dims(DIMS, index, 0);

		index[COEFF_DIM] = 3;

		const struct linop_s* linop_fftc = linop_fftc_create(DIMS, data->map_dims, FFT_FLAGS);

		md_copy(data->N, data->in_dims, data->in_tmp, src, CFL_SIZE);

		md_copy_block(DIMS, index, data->map_dims, data->tmp, data->in_dims, data->in_tmp, CFL_SIZE);

		linop_forward_unchecked(linop_fftc, data->tmp, data->tmp);

		md_copy_block(DIMS, index, data->in_dims, data->in_tmp, data->map_dims, data->tmp, CFL_SIZE);

		linop_free(linop_fftc);

		// Dump cfl

		char name[255] = { '\0' };

		sprintf(name, "current_map_%02d", data->counter);

		dump_cfl(name, data->N, data->in_dims, data->in_tmp);

		data->counter++;
	}
#endif

	// Allocate GPU memory
	complex float* r1scale_tmp;
	complex float* r2scale_tmp;
	complex float* m0scale_tmp;
	complex float* b1scale_tmp;


	if (data->use_gpu) {

#ifdef USE_CUDA
		r1scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
		r2scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
		m0scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
		b1scale_tmp = md_alloc_gpu(data->N, data->map_dims, CFL_SIZE);
#else
		assert(0);
#endif

	} else {

		r1scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
		r2scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
		m0scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
		b1scale_tmp = md_alloc(data->N, data->map_dims, CFL_SIZE);
	}


	long pos[data->N];

	md_set_dims(data->N, pos, 0);

	//-------------------------------------------------------------------
	// Copy necessary files from GPU to CPU
	//-------------------------------------------------------------------

	float scale[4];

	for (int i = 0; i < 4; i++) {

		scale[i] = data->moba_data->other.scale[i];

		if (0. == scale[i])
			scale[i] = 1.;
	}

	// R1
	pos[COEFF_DIM] = 0;
	const complex float* R1 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, r1scale_tmp, data->map_strs, R1, scale[0]);

	complex float* r1scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, r1scale, r1scale_tmp, CFL_SIZE);

	// M0
	pos[COEFF_DIM] = 1;
	const complex float* M0 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, m0scale_tmp, data->map_strs, M0, scale[1]);

	complex float* m0scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, m0scale, m0scale_tmp, CFL_SIZE);

	// R2
	pos[COEFF_DIM] = 2;
	const complex float* R2 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, r2scale_tmp, data->map_strs, R2, scale[2]);

	complex float* r2scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, r2scale, r2scale_tmp, CFL_SIZE);

	// B1
	pos[COEFF_DIM] = 3;
	const complex float* B1 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp, data->map_strs, B1, scale[3]);

	bloch_forw_alpha(data->linop_alpha, b1scale_tmp, data->tmp);	// freq -> pixel + smoothing!

	complex float* b1scale = md_alloc(data->N, data->map_dims, CFL_SIZE);
	md_copy(data->N, data->map_dims, b1scale, b1scale_tmp, CFL_SIZE);

	md_free(r1scale_tmp);
	md_free(r2scale_tmp);
	md_free(m0scale_tmp);
	md_free(b1scale_tmp);


	//Allocate Output CPU memory
	complex float* sig_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dr1_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dr2_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);
	complex float* dm0_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);
	complex float* db1_cpu = md_calloc(data->N, data->out_dims, CFL_SIZE);

	float fov_reduction_factor = data->moba_data->other.fov_reduction_factor;

	// Get start and end values of reduced F0V
	int xstart = round(data->map_dims[0] / 2. - fov_reduction_factor * data->map_dims[0] / 2.);
	int xend = round(xstart + fov_reduction_factor * data->map_dims[0]);
	int ystart = round(data->map_dims[1] / 2. - fov_reduction_factor * data->map_dims[1] / 2.);
	int yend = round(ystart + fov_reduction_factor * data->map_dims[1]);
	int zstart = round(data->map_dims[2] / 2. - fov_reduction_factor * data->map_dims[2] / 2.);
	int zend = round(zstart + fov_reduction_factor * data->map_dims[2]);

	// Solve rounding bug if fov_reduction_factor becomes to small, Change to round-up macro?!
	if (0 == zend)
		zend = 1;

	// debug_sim(&(data->moba_data.sim));

	#pragma omp parallel for collapse(3)
	for (int x = xstart; x < xend; x++) {
		for (int y = ystart; y < yend; y++) {
			for (int z = zstart; z < zend; z++) {

				//Calculate correct spatial position
				long spa_pos[DIMS];

				md_copy_dims(DIMS, spa_pos, data->map_dims);

				spa_pos[0] = x;
				spa_pos[1] = y;
				spa_pos[2] = z;

				long spa_ind = md_calc_offset(data->N, data->map_strs, spa_pos) / CFL_SIZE;

				//-------------------------------------------------------------------
				// Define simulation parameter
				//-------------------------------------------------------------------

                                // Extract external B1 value from input

				float b1s = 1.;

				if (NULL != data->b1) {

					b1s = crealf(data->b1[spa_ind]);

					if (safe_isnanf(b1s))
						b1s = 0.;
				}

                                // Copy simulation data

				struct sim_data sim_data = data->moba_data->sim;


                                sim_data.seq.rep_num = data->out_dims[TE_DIM];

                                sim_data.voxel.r1 = crealf(r1scale[spa_ind]);
				sim_data.voxel.r2 = crealf(r2scale[spa_ind]);
				sim_data.voxel.m0 = 1.;
				sim_data.voxel.b1 = b1s * crealf(b1scale[spa_ind]);

                                // debug_sim(&sim_data);
                                // debug_sim(&(data->moba_data.sim));

				//-------------------------------------------------------------------
				// Run simulation
				//-------------------------------------------------------------------

				float m[sim_data.seq.rep_num][3];
				float sa_r1[sim_data.seq.rep_num][3];
				float sa_r2[sim_data.seq.rep_num][3];
				float sa_m0[sim_data.seq.rep_num][3];
				float sa_b1[sim_data.seq.rep_num][3];

				bloch_simulation(&sim_data, data->slice, sim_data.seq.rep_num, &m, &sa_r1, &sa_r2, &sa_m0, &sa_b1);

				//-------------------------------------------------------------------
				// Copy simulation output to storage on CPU
				//-------------------------------------------------------------------

				long curr_pos[DIMS];
				md_copy_dims(DIMS, curr_pos, spa_pos);

				long position = 0;

				for (int j = 0; j < sim_data.seq.rep_num; j++) {

					curr_pos[TE_DIM] = j;

					position = md_calc_offset(data->N, data->out_strs, curr_pos) / CFL_SIZE;

					float a = 1.;

                                        assert(0 != sim_data.pulse.flipangle);

					// Scaling signal to 1
                                        //      -> Comparability to Look-Locker model

					if ((SEQ_FLASH == sim_data.seq.seq_type) || (SEQ_IRFLASH == sim_data.seq.seq_type))
						a = 1. / (sinf(sim_data.pulse.flipangle * M_PI / 180.) * expf(-sim_data.voxel.r2 * sim_data.seq.te));

                                        else if ((SEQ_BSSFP == sim_data.seq.seq_type) || (SEQ_IRBSSFP == sim_data.seq.seq_type))
                                                a = 1. / (sinf(sim_data.pulse.flipangle / 2. * M_PI / 180.) * expf(-sim_data.voxel.r2 * sim_data.seq.te));

					const float (*scale2)[4] = &data->moba_data->other.scale;

					// complex m0scale[spa_ind] adds scaling and phase to the signal
					// M = M_x + i M_y	and S = S_x + i S_y
					dr1_cpu[position] = a * (*scale2)[0] * m0scale[spa_ind] * (sa_r1[j][0] + sa_r1[j][1] * 1.i);
					dm0_cpu[position] = a * (*scale2)[1] * (sa_m0[j][0] + sa_m0[j][1] * 1.i);
					dr2_cpu[position] = a * (*scale2)[2] * m0scale[spa_ind] * (sa_r2[j][0] + sa_r2[j][1] * 1.i);
					db1_cpu[position] = a * (*scale2)[3] * m0scale[spa_ind] * (sa_b1[j][0] + sa_b1[j][1] * 1.i);
					sig_cpu[position] = a * m0scale[spa_ind] * (m[j][0] + m[j][1] * 1.i);
				}
			}
		}
	}

	md_free(r1scale);
	md_free(r2scale);
	md_free(m0scale);
	md_free(b1scale);

	debug_printf(DP_DEBUG3, "Copy data\n");

	//-------------------------------------------------------------------
	// Collect data of signal (potentially on GPU)
	//-------------------------------------------------------------------

	md_copy(data->N, data->out_dims, dst, sig_cpu, CFL_SIZE);

	md_free(sig_cpu);

	//-------------------------------------------------------------------
	// Collect data of derivatives in single array
	//-------------------------------------------------------------------

	md_clear(data->N, data->der_dims, data->derivatives, CFL_SIZE);

	md_set_dims(data->N, pos, 0);

	pos[COEFF_DIM] = 0; // R1
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dr1_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 1; // M0
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dm0_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 2; // R2
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, dr2_cpu, CFL_SIZE);

	pos[COEFF_DIM] = 3; // B1
	md_copy_block(data->N, pos, data->der_dims, data->derivatives, data->out_dims, db1_cpu, CFL_SIZE);

	md_free(dr1_cpu);
	md_free(dr2_cpu);
	md_free(dm0_cpu);
	md_free(db1_cpu);

	double totaltime = timestamp() - starttime;

	debug_printf(DP_DEBUG2, "Time = %.2f s\n", totaltime);
}


static void bloch_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	debug_printf(DP_DEBUG3, "Start Derivative\n");

	struct blochfun_s* data = CAST_DOWN(blochfun_s, _data);

	// Transform B1 map component from freq to pixel domain

	long pos[data->N];
	md_set_dims(data->N, pos, 0);
	pos[COEFF_DIM] = 3;

	md_clear(data->N, data->in_dims, data->in_tmp, CFL_SIZE);
	md_copy(data->N, data->in_dims, data->in_tmp, src, CFL_SIZE);

	md_copy_block(data->N, pos, data->map_dims, data->tmp, data->in_dims, data->in_tmp, CFL_SIZE);
	bloch_forw_alpha(data->linop_alpha, data->tmp, data->tmp); // freq -> pixel + smoothing!
	md_copy_block(data->N, pos, data->in_dims, data->in_tmp, data->map_dims, data->tmp, CFL_SIZE);

	// Estimate derivative of operator
	// FIXME: replace by linop to use automatic gram matrix calculation
	md_clear(data->N, data->out_dims, dst, CFL_SIZE);
	md_ztenmul(data->N, data->out_dims, dst, data->der_dims, data->derivatives, data->in_dims, data->in_tmp);
}



static void bloch_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	debug_printf(DP_DEBUG3, "Start Adjoint\n");

	struct blochfun_s* data = CAST_DOWN(blochfun_s, _data);

	// Estimate adjoint derivative of operator
	// FIXME: replace by linop to use automatic gram matrix calculation
	md_clear(data->N, data->in_dims, data->in_tmp, CFL_SIZE);
	md_zfmacc2(data->N, data->der_dims, data->in_strs, data->in_tmp, data->out_strs, src, data->der_strs, data->derivatives);

	// Transform B1 map component from pixel to freq domain

	long pos[data->N];
	md_set_dims(data->N, pos, 0);
	pos[COEFF_DIM] = 3;

	md_copy_block(data->N, pos, data->map_dims, data->tmp, data->in_dims, data->in_tmp, CFL_SIZE);

	bloch_back_alpha(data->linop_alpha, data->tmp, data->tmp); // pixel -> freq + smoothing!

	md_copy_block(data->N, pos, data->in_dims, data->in_tmp, data->map_dims, data->tmp, CFL_SIZE);

	md_copy(data->N, data->in_dims, dst, data->in_tmp, CFL_SIZE);
}


static void bloch_del(const nlop_data_t* _data)
{
	struct blochfun_s* data = CAST_DOWN(blochfun_s, _data);

	md_free(data->derivatives);
	md_free(data->tmp);
	md_free(data->in_tmp);
	md_free(data->weights);

	xfree(data->der_dims);
	xfree(data->map_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->der_strs);
	xfree(data->map_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);

	linop_free(data->linop_alpha);

	xfree(data);
}


struct nlop_s* nlop_bloch_create(int N, const long der_dims[N], const long map_dims[N], const long out_dims[N], const long in_dims[N],
			const complex float* b1, const complex float* slice, const struct moba_conf_s* config, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct blochfun_s, data);
	SET_TYPEID(blochfun_s, data);

	PTR_ALLOC(long[N], derdims);
	md_copy_dims(N, *derdims, der_dims);
	data->der_dims = *PTR_PASS(derdims);

	PTR_ALLOC(long[N], allstr);
	md_calc_strides(N, *allstr, der_dims, CFL_SIZE);
	data->der_strs = *PTR_PASS(allstr);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nmstr);
	md_calc_strides(N, *nmstr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nmstr);

	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr);

	PTR_ALLOC(long[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

	PTR_ALLOC(long[N], nistr);
	md_calc_strides(N, *nistr, in_dims, CFL_SIZE);
	data->in_strs = *PTR_PASS(nistr);

	data->N = N;

	data->derivatives = my_alloc(N, der_dims, CFL_SIZE);
	data->tmp = my_alloc(N, map_dims, CFL_SIZE);
	data->in_tmp = my_alloc(N, in_dims, CFL_SIZE);


	data->moba_data = config;

	data->b1 = b1;
	data->slice = slice;

	data->use_gpu = use_gpu;

	data->counter = 0;

	// Smoothness penalty for alpha map: Sobolev norm

	long w_dims[N];
	md_select_dims(N, FFT_FLAGS, w_dims, map_dims);

	complex float* weights = md_alloc(N, w_dims, CFL_SIZE);
	noir_calc_weights(440., 20., w_dims, weights);
	data->weights = weights;

	const struct linop_s* linop_wghts = linop_cdiag_create(N, map_dims, FFT_FLAGS, data->weights);
	const struct linop_s* linop_ifftc = linop_ifftc_create(N, map_dims, FFT_FLAGS);

	data->linop_alpha = linop_chain(linop_wghts, linop_ifftc);

	linop_free(linop_wghts);
	linop_free(linop_ifftc);

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), bloch_fun, bloch_der, bloch_adj, NULL, NULL, bloch_del);
}
