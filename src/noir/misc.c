
#include <stdlib.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mri.h"

#include "misc.h"


struct ds_s {

	long dims_full[DIMS];
	long dims_singleFrame[DIMS];
	long dims_output[DIMS];
	long dims_output_singleFrame[DIMS];


	long strs_full[DIMS];
	long strs_singleFrame[DIMS];
	long strs_output[DIMS];
	long strs_output_singleFrame[DIMS];
};


// Initialize dimensions and strides
static void ds_init(struct ds_s* in, size_t size)
{
	md_select_dims(DIMS, ~TIME_FLAG, in->dims_singleFrame, in->dims_full);
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG, in->dims_output_singleFrame, in->dims_full);
	md_select_dims(DIMS, FFT_FLAGS|SLICE_FLAG|TIME_FLAG, in->dims_output, in->dims_full);

	md_calc_strides(DIMS, in->strs_full, in->dims_full, size);
	md_calc_strides(DIMS, in->strs_singleFrame, in->dims_singleFrame, size);
	md_calc_strides(DIMS, in->strs_output, in->dims_output, size);
	md_calc_strides(DIMS, in->strs_output_singleFrame, in->dims_output_singleFrame, size);
}




// Normalization of PSF and scaling of k-space
void scale_psf_k(const long pat_dims[DIMS], complex float* pattern, const long ksp_dims[DIMS], complex float* kspace_data, const long trj_dims[DIMS], complex float* traj)
{
	/* PSF
	* Since for each frame we can have a different number of spokes,
	* some spoke-lines are empty in certain frames. To ensure
	* adequate normalization we have to calculate how many spokes are there
	* in each frame and build the inverse
	*
	* Basic idea:
	* Summation of READ_DIM and PHS1_DIM:
	* If the result is zero the spoke-line was empty
	*/

	long traj_dims2[DIMS]; // Squashed trajectory array
	md_copy_dims(DIMS, traj_dims2, trj_dims);
	traj_dims2[READ_DIM] = 1;
	traj_dims2[PHS1_DIM] = 1;

	complex float* traj2 = md_alloc(DIMS, traj_dims2, CFL_SIZE);
	md_zrss(DIMS, trj_dims, READ_FLAG|PHS1_FLAG, traj2, traj);
	md_zdiv(DIMS, traj_dims2, traj2, traj2, traj2); // Normalize each non-zero element to one

	/* Sum the ones (non-zero elements) to get
	* number of spokes in each cardiac frame
	*/
	struct ds_s* no_spf_s = (struct ds_s*) malloc(sizeof(struct ds_s));
	md_copy_dims(DIMS, no_spf_s->dims_full, traj_dims2);
	no_spf_s->dims_full[PHS2_DIM] = 1;
	ds_init(no_spf_s, CFL_SIZE);

	complex float* no_spf = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
	md_clear(DIMS, no_spf_s->dims_full, no_spf, CFL_SIZE);
	md_zrss(DIMS, traj_dims2, PHS2_FLAG, no_spf, traj2);
	md_zspow(DIMS, no_spf_s->dims_full, no_spf, no_spf, 2); // no_spf contains the number of spokes in each frame and partition

	// Inverse (inv_no_spf contains inverse of number of spokes in each frame/partition)
	complex float* inv_no_spf = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
	md_zfill(DIMS, no_spf_s->dims_full, inv_no_spf, 1.);
	md_zdiv(DIMS, no_spf_s->dims_full, inv_no_spf, inv_no_spf, no_spf);


	long pat_strs[DIMS];
	md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

	// Multiply PSF
	md_zmul2(DIMS, pat_dims, pat_strs, pattern, pat_strs, pattern, no_spf_s->strs_full, inv_no_spf);
	// 	dump_cfl("PSF", DIMS, pat_s->dims_full, pattern);

	/* k
	 * Scaling of k-space (depending on total [= all partitions] number of spokes per frame)
	 * Normalization is not performed here)
	 */

	long no_spf_s_dims_singlePart[DIMS];
	long no_spf_s_dims_singleFramePart[DIMS];
	md_select_dims(DIMS, ~SLICE_FLAG, no_spf_s_dims_singlePart, no_spf_s->dims_full);
	md_select_dims(DIMS, ~(TIME_FLAG|SLICE_FLAG), no_spf_s_dims_singleFramePart, no_spf_s->dims_full);

	long no_spf_s_strs_singlePart[DIMS];
	long no_spf_s_strs_singleFramePart[DIMS];
	md_calc_strides(DIMS, no_spf_s_strs_singlePart, no_spf_s_dims_singlePart, CFL_SIZE);
	md_calc_strides(DIMS, no_spf_s_strs_singleFramePart, no_spf_s_dims_singleFramePart, CFL_SIZE);


	// Sum spokes in all partitions
	complex float* no_spf_tot = md_alloc(DIMS, no_spf_s_dims_singlePart, CFL_SIZE);
	md_zsum(DIMS, no_spf_s->dims_full, SLICE_FLAG, no_spf_tot, no_spf);

	// Extract first frame
	complex float* no_sp_1stFrame_tot = md_alloc(DIMS, no_spf_s_dims_singleFramePart, CFL_SIZE);
	long posF[DIMS] = { 0 };
	md_copy_block(DIMS, posF, no_spf_s_dims_singleFramePart, no_sp_1stFrame_tot, no_spf_s_dims_singlePart, no_spf_tot, CFL_SIZE);

	complex float* ksp_scaleFactor = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
	md_clear(DIMS, no_spf_s->dims_full, ksp_scaleFactor, CFL_SIZE);

	complex float* inv_no_spf_tot = md_alloc(DIMS, no_spf_s->dims_full, CFL_SIZE);
	md_zfill(DIMS, no_spf_s_dims_singlePart, inv_no_spf_tot, 1.);
	md_zdiv(DIMS, no_spf_s_dims_singlePart, inv_no_spf_tot, inv_no_spf_tot, no_spf_tot);
	md_zmul2(DIMS, no_spf_s->dims_full, no_spf_s->strs_full, ksp_scaleFactor, no_spf_s_strs_singlePart, inv_no_spf_tot, no_spf_s_strs_singleFramePart, no_sp_1stFrame_tot);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);
	md_zmul2(DIMS, ksp_dims, ksp_strs, kspace_data, ksp_strs, kspace_data, no_spf_s->strs_full, ksp_scaleFactor);

	free(no_spf_s);
	md_free(no_spf_tot);
	md_free(inv_no_spf_tot);
	md_free(ksp_scaleFactor);
	md_free(no_sp_1stFrame_tot);

	md_free(traj2);
	md_free(no_spf);
	md_free(inv_no_spf);
}




void postprocess(const long dims[DIMS], bool normalize,
			const long sens_strs[DIMS], const complex float* sens,
			const long img_strs[DIMS], const complex float* img,
			const long img_output_dims[DIMS], const long img_output_strs[DIMS], complex float* img_output)
{
	if (md_calc_size(3, img_output_dims) != md_calc_size(3, dims)) {

		long img_output2_dims[DIMS];
		md_copy_dims(DIMS, img_output2_dims, img_output_dims);
		md_copy_dims(3, img_output2_dims, dims);

		long img_output2_strs[DIMS];
		md_calc_strides(DIMS, img_output2_strs, img_output2_dims, CFL_SIZE);

		complex float* tmp = md_alloc(DIMS, img_output2_dims, CFL_SIZE);

		postprocess(dims, normalize, sens_strs, sens, img_strs, img, img_output2_dims, img_output2_strs, tmp);

		md_resize_center(DIMS, img_output_dims, img_output, img_output2_dims, tmp, CFL_SIZE);

		md_free(tmp);
		return;
	}

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

	long ksp_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, ksp_dims, dims);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	long ksp_strs[DIMS];
	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);


	int nmaps = dims[MAPS_DIM];
	bool combine = (1 == img_output_dims[MAPS_DIM]);


	// image output
	if (normalize) {

		complex float* buf = md_alloc(DIMS, dims, CFL_SIZE);
		md_clear(DIMS, dims, buf, CFL_SIZE);

		if (combine) {

			md_zfmac2(DIMS, dims, ksp_strs, buf, img_strs, img, sens_strs, sens);
			md_zrss(DIMS, ksp_dims, COIL_FLAG, img_output, buf);

		} else {

			md_zfmac2(DIMS, dims, strs, buf, img_strs, img, sens_strs, sens);
			md_zrss(DIMS, dims, COIL_FLAG, img_output, buf);
		}

//		md_zmul2(DIMS, img_output_dims, img_output_strs, img_output, img_output_strs, img_output, msk_strs, mask);

		if ((1 == nmaps) || !combine) {

			//restore phase
			md_zphsr(DIMS, img_output_dims, buf, img);
			md_zmul(DIMS, img_output_dims, img_output, img_output, buf);
		}

		md_free(buf);

	} else {

		if (combine) {

			// just sum up the map images
			md_clear(DIMS, img_output_dims, img_output, CFL_SIZE);
			md_zaxpy2(DIMS, img_dims, img_output_strs, img_output, 1., img_strs, img);

		} else { /*!normalize && !combine */

			md_copy(DIMS, img_output_dims, img_output, img, CFL_SIZE);
		}
	}
}


