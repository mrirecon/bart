/*
 * Copyright 2013-2015 The Regents of the University of California.
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
struct dfwavelet_plan_s {
  int use_gpu;

  unsigned int state;

  int numdims;
  long* imSize; // Input Image Size
  long numPixel; // Number of image pixels
  long numCoeff; // Number of wavelet coefficients
  data_t* res; // Resolution
  scalar_t* noiseAmp; // Noise amplification for each subband
  scalar_t percentZero;

  long* minSize; // Minimum size for the scaling subband
  long numCoarse;
  long* waveSizes; // Contains all wavelet subband sizes
  int numLevels;
  int* randShift;

	_Bool randshift;

  // Filter parameters
  int filterLen;
  scalar_t* lod0;
  scalar_t* hid0;
  scalar_t* lor0;
  scalar_t* hir0;
  scalar_t* lod1;
  scalar_t* hid1;
  scalar_t* lor1;
  scalar_t* hir1;

};


