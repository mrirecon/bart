
#include <stdio.h>
#include <complex.h>

#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "simu/signals.h"

#include "utest.h"

static bool test_looklocker(void)
{
	struct signal_model data = signal_looklocker_defaults;

	float echos = 200;

	data.short_tr_LL_approx = true;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = echos;

	complex float* signal = md_alloc(DIMS, dims, CFL_SIZE);

	looklocker_model(&data, echos, signal);

	if (1E-5 < (cabsf(signal[0]) - 1.))
		return 0;

	if (1E-5 < (cabsf(signal[100]) - 0.027913))
		return 0;

	if (1E-5 < (cabsf(signal[199]) - 0.213577))
		return 0;

	md_free(signal);

	return 1;
}

UT_REGISTER_TEST(test_looklocker);


static bool test_looklocker_long_TR(void)
{
	struct signal_model data = signal_looklocker_defaults;

	float echos = 200;

	data.short_tr_LL_approx = true;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = echos;

	complex float* signal = md_alloc(DIMS, dims, CFL_SIZE);

	// Test extreme case for very short TR

	data.tr = 0.001;

	looklocker_model(&data, echos, signal);

	if (1E-5 < (cabsf(signal[0]) - 1.))
		return 0;

	if (1E-5 < (cabsf(signal[100]) - 0.279089))
		return 0;

	if (1E-5 < (cabsf(signal[199]) - 0.035142))
		return 0;

	// Turn off short TR assumption and check against extreme case

	data.short_tr_LL_approx = false;

	looklocker_model(&data, echos, signal);

	if (1E-5 < (cabsf(signal[0]) - 1.))
		return 0;

	if (1E-5 < (cabsf(signal[100]) - 0.279089))
		return 0;

	if (1E-5 < (cabsf(signal[199]) - 0.035142))
		return 0;

	// Test long TR case next

	data.tr = 0.015;

	looklocker_model(&data, echos, signal);

	if (1E-5 < (cabsf(signal[0]) - 1.))
		return 0;

	if (1E-5 < (cabsf(signal[100]) - 0.473335))
		return 0;

	if (1E-5 < (cabsf(signal[199]) - 0.596685))
		return 0;

	md_free(signal);

	return 1;
}

UT_REGISTER_TEST(test_looklocker_long_TR);



static bool test_IR_bSSFP(void)
{
	struct signal_model data = signal_IR_bSSFP_defaults;

	float echos = 200;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = echos;

	complex float* signal = md_alloc(DIMS, dims, CFL_SIZE);

	IR_bSSFP_model(&data, echos, signal);

	if (1E-5 < (cabsf(signal[0]) - 0.382683))
		return 0;

	if (1E-5 < (cabsf(signal[100]) - 0.036060))
		return 0;

	if (1E-5 < (cabsf(signal[199]) - 0.085378))
		return 0;

// 	for (unsigned int i = 0; i < echos; i++)
// 		printf("signal[%d]: %f\n", i, cabsf(signal[i]));

	md_free(signal);

	return 1;
}

UT_REGISTER_TEST(test_IR_bSSFP);


static bool test_buxton(void)
{
	struct signal_model data = signal_buxton_defaults;

	float timepoints = 200;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = timepoints;

	complex float* signal = md_alloc(DIMS, dims, CFL_SIZE);

	buxton_model(&data, timepoints, signal);

	if (1E-5 < (cabsf(signal[0]) - 0.0))
		return 0;

	if (1E-5 < (cabsf(signal[50]) - 0.0))
		return 0;

	if (1E-5 < (cabsf(signal[100]) - 0.005850))
		return 0;

	if (1E-5 < (cabsf(signal[199]) - 0.012706))
		return 0;

	md_free(signal);

	return 1;
}

UT_REGISTER_TEST(test_buxton);



static bool test_buxton_pulsed(void)
{
	struct signal_model data = signal_buxton_pulsed;

	float timepoints = 200;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = timepoints;

	complex float* signal = md_alloc(DIMS, dims, CFL_SIZE);

	buxton_model(&data, timepoints, signal);

	if (1E-5 < (cabsf(signal[0]) - 0.0))
		return 0;

	if (1E-5 < (cabsf(signal[50]) - 0.0))
		return 0;

	if (1E-5 < (cabsf(signal[100]) - 0.005766))
		return 0;

	if (1E-5 < (cabsf(signal[199]) - 0.004581))
		return 0;

	md_free(signal);

	return 1;
}

UT_REGISTER_TEST(test_buxton_pulsed);


static bool test_buxton_after_labeling(void)
{
	struct signal_model data = signal_buxton_defaults;
	data.acquisition_only = true;

	float timepoints = 200;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[TE_DIM] = timepoints;

	complex float* signal = md_alloc(DIMS, dims, CFL_SIZE);

	buxton_model(&data, timepoints, signal);

	if (1E-5 < (cabsf(signal[0]) - 0.014021))
		return 0;

	if (1E-5 < (cabsf(signal[50]) - 0.009755))
		return 0;

	if (1E-5 < (cabsf(signal[100]) - 0.006788))
		return 0;

	if (1E-5 < (cabsf(signal[199]) - 0.003310))
		return 0;

	md_free(signal);

	return 1;
}

UT_REGISTER_TEST(test_buxton_after_labeling);
