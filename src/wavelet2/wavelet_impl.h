/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
 

/**
 * data_t is the type for input/output data, can be float/double or _Complex float/double.
 */
typedef _Complex float data_t;
/**
 * scalar_t is the type for filter and scalers, can be float/double
 */
typedef float scalar_t;


/**
 * Wavelet Plan struct
 *
 * @param imSize Input Image Size
 * @param numPixel Number of image pixels
 * @param numCoeff Number of wavelet coefficients
 * @param trDims Which dimensions we do wavelet transform
 * @param minSize_tr Minimum size for the scaling subband
 * @param waveSizes_tr Contains all wavelet subband sizes
 *
 */
struct wavelet_plan_s {
	float lambda;
	int use_gpu;
	bool randshift;
	unsigned int flags;

	// state of random number generator
	unsigned int state;

	// Input/Output parameters (5d) [X,Y,Z,Coils,Echos]

	int numdims;
	long* imSize; // Input Image Size
	long numPixel; // Number of image pixels
	long numCoeff; // Number of wavelet coefficients
	long batchSize; // Batch Size, eg Coils*Echos or Z*Coils*Echos

	// Parameters for each wavelet transform, can only be 2d or 3d
	int numdims_tr;
	long* trDims; // Which dimensions we do wavelet transform
	long* imSize_tr;
	long* minSize_tr; // Minimum size for the scaling subband
	long numPixel_tr;
	long numCoeff_tr;
	long numCoarse_tr;
	long* waveSizes_tr; // Contains all wavelet subband sizes
	int numLevels_tr;
	long* randShift_tr;

	// Temp memory
	data_t* tmp_mem_tr;

	// Filter parameters
	int filterLen;
	const scalar_t* lod;
	const scalar_t* hid;
	const scalar_t* lor;
	const scalar_t* hir;

};

#ifdef __cplusplus
extern "C" {
#endif
	extern void circshift(struct wavelet_plan_s* plan, data_t *data);
	extern void circunshift(struct wavelet_plan_s* plan, data_t *data);
#ifdef __cplusplus
}
#endif



