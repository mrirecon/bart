/* Copyright 2022. TU Graz. Insitute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <math.h>

#include "misc/misc.h"
#include "misc/mri.h"

#include "simu/pulse.h"
#include "simu/simulation.h"

#include "utest.h"

static bool test_sinc_integral(void)
{
        struct simdata_pulse pulse = simdata_pulse_defaults;

        sinc_pulse_init(&pulse, 0., 0.001, 180., 0., 4., 0.46);

        if ((M_PI - sinc_integral(&pulse)) > 10E-7)
                return 0;
        else
                return 1;
}

UT_REGISTER_TEST(test_sinc_integral);


static bool test_sinc_integral2(void)
{
        struct simdata_pulse pulse = simdata_pulse_defaults;

        sinc_pulse_init(&pulse, 0., 0.001, 180., 0., 4., 0.46);


        // Estimate integral with trapezoidal rule

        float dt = 10E-8;

        float integral = 0.5 * pulse_sinc(&pulse, pulse.rf_start) * dt;

        float t = pulse.rf_start + dt;

        while (t < pulse.rf_end) {

                integral += pulse_sinc(&pulse, t) * dt;

                t += dt;
        }

        integral += 0.5 * pulse_sinc(&pulse, pulse.rf_end) * dt;

        float error = fabs(M_PI - integral);

        // debug_printf(DP_WARN, "Estimated Integral: %f,\t Error: %f\n", integral, error);

        if (error > 10E-5)
                return 0;
        else
        return 1;
}

UT_REGISTER_TEST(test_sinc_integral2);


// Test Accuracy of on-resonant pulse
//      1. Execute pulse for various angles and durations
//      2. Compare final magnetization to nominal angle set for the pulse
static bool test_rf_pulse_ode(void)
{
	long dim[DIMS] = { [0 ... DIMS - 1] = 1 };

        enum { N = 3 };              // Number of dimensions (x, y, z)
	enum { P = 4 };              // Number of parameters with estimated derivative (Mxy, R1, R2, B1)

	dim[READ_DIM] = 10;
	dim[PHS1_DIM] = 10;

        // RF duration
	float tmin = 0.0001;
	float tmax = 0.1;

        // Nominal FA
	float amin = 0.;
	float amax = 180.;

	for (int i = 0; i < dim[READ_DIM]; i++ )
		for (int j = 0; j < dim[PHS1_DIM]; j++ ) {

			float trf = (tmin + (float)i/((float)dim[READ_DIM] - 1.) * (tmax - tmin));
			float angle = (amin + (float)j/((float)dim[PHS1_DIM] - 1.) * (amax - amin));

                        // Define sequence characteristics
			struct sim_data data;

			data.seq = simdata_seq_defaults;
			data.seq.seq_type = SEQ_IRBSSFP;
			data.seq.tr = 10.;
			data.seq.te = 5.;
			data.seq.rep_num = 1;
			data.seq.spin_num = 1;

			data.voxel = simdata_voxel_defaults;
			data.voxel.r1 = 0.;
			data.voxel.r2 = 0.;
			data.voxel.m0 = 1;
			data.voxel.w = 0;

			data.pulse = simdata_pulse_defaults;
			data.pulse.flipangle = angle;
			data.pulse.rf_end = trf;

			data.grad = simdata_grad_defaults;
			data.tmp = simdata_tmp_defaults;


                        // Prepare pulse
			sinc_pulse_init(&data.pulse, 0., trf, angle, 0., 4., 0.46);

			float xp[4][3] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };

			float h = 10E-5;
			float tol = 10E-6;

                        // Run pulse
			rf_pulse(&data, h, tol, N, P, xp, NULL);


                        // Compare result to nominal FA
			float sim_angle = 0.;

                        // FA <= 90°
			if (xp[0][2] >= 0) {

				if (data.voxel.r1 != 0 && data.voxel.r2 != 0)   // relaxation case
					sim_angle = asinf(xp[0][1] / data.voxel.m0) / M_PI * 180.;
				else
					sim_angle = asinf(xp[0][1] / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2])) / M_PI * 180.;
			}
                        // FA > 90°
			else {
				if (data.voxel.r1 != 0 && data.voxel.r2 != 0)   // relaxation case
					sim_angle = acosf(fabs(xp[0][1]) / data.voxel.m0) / M_PI * 180. + 90.;
				else
					sim_angle = acosf(fabs(xp[0][1]) / sqrtf(xp[0][0]*xp[0][0]+xp[0][1]*xp[0][1]+xp[0][2]*xp[0][2])) / M_PI * 180. + 90.;
			}

			float err = fabs(data.pulse.flipangle - sim_angle);

			if (err > 10E-4) {

				debug_printf(DP_WARN, "Error for test_rf_pulse_ode\n see -> utests/test_pulse.c\n");
				return 0;
			}
		}

	return 1;
}

UT_REGISTER_TEST(test_rf_pulse_ode);



static bool test_hypsec_rf_pulse_ode(void)
{
        enum { N = 3 };              // Number of dimensions (x, y, z)
	enum { P = 4 };              // Number of parameters with estimated derivative (Mxy, R1, R2, B1)

        struct sim_data data;

        data.seq = simdata_seq_defaults;
        data.seq.seq_type = SEQ_FLASH;
        data.seq.tr = 0.001;
        data.seq.te = 0.001;
        data.seq.rep_num = 1;
        data.seq.spin_num = 1;

        data.voxel = simdata_voxel_defaults;
        data.voxel.r1 = 0.;
        data.voxel.r2 = 0.;
        data.voxel.m0 = 1;
        data.voxel.w = 0;

        data.pulse = simdata_pulse_defaults;
        data.pulse.flipangle = 0.;      // Turn off flipangle -> do not influence inversion efficiency
        data.pulse.rf_end = 0.01;

        // Hyperbolic Secant Characteristics
        data.pulse.hs = hs_pulse_defaults;
        data.pulse.hs.on = true;
        data.pulse.hs.duration = data.pulse.rf_end;

        data.grad = simdata_grad_defaults;
        data.tmp = simdata_tmp_defaults;

        float xp[P][N] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };

        float h = 10E-5;
        float tol = 0.005; // >99.5% inversion efficiency

        rf_pulse(&data, h, tol, N, P, xp, NULL);

        // bart_printf("%f, %f, %f\n", xp[0][0], xp[0][1], xp[0][2]);

        UT_ASSERT(fabs(xp[0][2] + 1.) < tol);

	return 1;
}

UT_REGISTER_TEST(test_hypsec_rf_pulse_ode);
