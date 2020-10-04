/* Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019-2020 Martin Uecker <muecker@gwdg.de>
 */

#include "polygon.h"


// inside polygon (Sunday Dan)

int polygon_winding_number(int N, const double pg[N][2], const double p[2])
{
	int n = 0;

	for (int i = 0; i < N; i++) {

		double o = (pg[(i + 1) % N][0] - pg[i][0]) * (p[1] - pg[i][1])
				- (p[0] - pg[i][0]) * (pg[(i + 1) % N][1] - pg[i][1]); 

		if (pg[i][1] <= p[1]) {

			if (   (pg[(i + 1) % N][1] > p[1])
			    && (o > 0.))	// left
				n++;

		} else {

			if (   (pg[(i + 1) % N][1] <= p[1])
			    && (o < 0.)) 	// right
				n--;
		}
	}

	return n;
}


double polygon_area(int N, const double pos[N][2])
{
       double sum = 0.;

       for (int i = 0; i < N; i++) {

               const double p[2][2] = {
                       { pos[(i + 0) % N][0], pos[(i + 0) % N][1] },
                       { pos[(i + 1) % N][0], pos[(i + 1) % N][1] },
               };

               sum += 0.5 * (p[0][0] * p[1][1] - p[1][0] * p[0][1]);
       }

       return sum;
}



