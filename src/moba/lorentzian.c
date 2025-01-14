/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/mri.h"

#include "num/multind.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/snlop.h"
#include "nlops/smath.h"

#include "lorentzian.h"


// calculate the lorentzian term for a pool
static arg_t lorentzian(arg_t amplitude, arg_t width, arg_t omegai, arg_t omega)
{
        // amplitude * (width/2)^2 / ((width/2)^2 + (omega - omegai)^2))

        // (omega - omegai)^2
	arg_t domega = snlop_sub(omega, omegai);
        arg_t domega2 = snlop_mul(domega, domega, 0);
        // (width / 2)^2
	arg_t width_term = snlop_scale(width, 0.5);
        arg_t width_term2 = snlop_mul(width_term, width_term, 0);
        // (omega - omegai)^2 / (width / 2)^2
        arg_t denominator = snlop_div(domega2, width_term2, 0);
	denominator = snlop_add_F(denominator, snlop_scalar(1.));
        arg_t fraction = snlop_div_F(amplitude, denominator, 0);
        return fraction;
}

// Function to reconstruct the signal with given pool parameters
const struct nlop_s* nlop_lorentzian_multi_pool_create(int N, const long signal_dims[N],
                const long param_dims[N], const long omega_dims[N], const complex float* omega)
{
        arg_t omega_arg = snlop_const(N, omega_dims, omega, "omega");

        long map_dims[DIMS];
        md_select_dims(DIMS, FFT_FLAGS, map_dims, signal_dims);

        int n_params = param_dims[COEFF_DIM]; 

        arg_t lp_term = snlop_scalar(1);

        // M0 exists once, every pool has 3 parameters
        if ((n_params < 1) || ((n_params - 1) % 3 != 0)) {
                error("amount of parameters is incorrect\n");
        }

        int n_pools = (n_params - 1)/3;
        arg_t amplitudes[n_pools];
        arg_t widths[n_pools];
        arg_t omegas[n_pools];

        arg_t args[n_params];
        int idx = 0;
        args[idx++] = snlop_input(N, map_dims, "M0");

        for (int i = 0; i < n_pools; i++) {

                amplitudes[i] = snlop_input(N, map_dims, "amplitude");
                widths[i] = snlop_input(N, map_dims, "width");
                omegas[i] = snlop_input(N, map_dims, "omegai");

                arg_t real_amplitude = snlop_real(amplitudes[i]);
                arg_t real_width = snlop_real(widths[i]);
                arg_t real_omegai = snlop_real(omegas[i]);

                // calculate the lorentzian for each pool
                arg_t signal = lorentzian(real_amplitude, real_width, real_omegai, omega_arg);
                lp_term = snlop_sub(lp_term, signal);

                args[idx++] = amplitudes[i];
                args[idx++] = widths[i];
                args[idx++] = omegas[i];
        }

        arg_t vFitValues = snlop_mul(args[0], lp_term, 0);

        const struct nlop_s* ret = nlop_from_snlop_F(
                snlop_from_arg(vFitValues),
                1,
                (arg_t[1]) { vFitValues },
                n_params,
                args
        );

        // Stack inputs for the final result
        for (int i = 0; i < n_params - 1; i++)
                ret = nlop_stack_inputs_F(ret, 0, 1, COEFF_DIM);

        return ret;
}