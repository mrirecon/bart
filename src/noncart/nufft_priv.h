



/**
 *
 * NUFFT internal data structure
 *
 */
struct nufft_data {

	INTERFACE(linop_data_t);

	struct nufft_conf_s conf;	///< NUFFT configuration structure

	unsigned int N;			///< Number of dimension
	unsigned long flags;

	const complex float* linphase;	///< Linear phase for pruned FFT
	const complex float* traj;	///< Trajectory
	const complex float* roll;	///< Roll-off factor
	const complex float* psf;	///< Point-spread function (2x size)
	const complex float* fftmod;	///< FFT modulation for centering
	const complex float* weights;	///< Weights, ex, density compensation
	const complex float* basis;
#ifdef USE_CUDA
	const complex float* linphase_gpu;
	const complex float* psf_gpu;
	complex float* grid_gpu;
#endif
	complex float* grid;		///< Oversampling grid

	float width;			///< Interpolation kernel width
	double beta;			///< Kaiser-Bessel beta parameter

	const struct linop_s* fft_op;	///< FFT operator

	long* ksp_dims;			///< Kspace dimension
	long* cim_dims;			///< Coil image dimension
	long* cml_dims;			///< Coil + linear phase dimension
	long* img_dims;			///< Image dimension
	long* trj_dims;			///< Trajectory dimension
	long* lph_dims;			///< Linear phase dimension
	long* psf_dims;			///< Point spread function dimension
	long* wgh_dims;			///< Weights dimension
	long* bas_dims;
	long* out_dims;
	long* ciT_dims;			///< Coil image dimension
	long* cmT_dims;			///< Coil + linear phase dimension

	//!
	long* cm2_dims;			///< 2x oversampled coil image dimension

	long* ksp_strs;
	long* cim_strs;
	long* cml_strs;
	long* img_strs;
	long* trj_strs;
	long* lph_strs;
	long* psf_strs;
	long* wgh_strs;
	long* bas_strs;
	long* out_strs;

	const struct linop_s* cfft_op;   ///< Pcycle FFT operator
	unsigned int cycle;
};




