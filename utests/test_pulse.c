/* Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <math.h>

#include "misc/misc.h"

#include "simu/pulse.h"

#include "utest.h"

static bool test_sinc_integral(void)
{

struct simdata_pulse pulse = simdata_pulse_defaults;

pulse_create(&pulse, 0., 0.001, 180., 0., 4., 0.46);

        if ((M_PI - sinc_integral(&pulse)) > 10E-7)
                return 0;
        else
                return 1;
}

UT_REGISTER_TEST(test_sinc_integral);


static bool test_sinc_integral2(void)
{
        struct simdata_pulse pulse = simdata_pulse_defaults;

        pulse_create(&pulse, 0., 0.001, 180., 0., 4., 0.46);


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