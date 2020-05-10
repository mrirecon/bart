
#ifndef DIMS
#define DIMS 16
#endif

extern void scale_psf_k(const long pat_dims[DIMS],
			_Complex float* pattern,
			const long ksp_dims[DIMS],
			_Complex float* kspace_data,
			const long trj_dims[DIMS],
			_Complex float* traj);

extern void postprocess(const long dims[DIMS], bool normalize,
			const long sens_strs[DIMS], const complex float* sens,
			const long img_strs[DIMS], const complex float* img,
			const long img_output_dims[DIMS], const long img_output_strs[DIMS], complex float* img_output);

