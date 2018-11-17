#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <complex.h>
#include <mkl.h>
#include <math.h>
#include <omp.h>

#include <sys/time.h>

void mysvthresh(complex float *buf, MKL_INT _M, MKL_INT _N, float *s,
                complex float *u, complex float *vt, complex float *work,
                MKL_INT lwork, float *rwork, MKL_INT info, float lambda) {
  complex float alpha = 1.0f;
  complex float beta = 0.0f;

  csyrk("L", "T", &_N, &_M, &alpha, buf, &_M, &beta, vt, &_N);

  float s_upperbound = 0;
  for (int i = 0; i < _N; i++) {
    float s = 0;
    for (int j = 0; j < _N; j++) {
      int row = (i > j ? i : j);
      int col = (i < j ? i : j);
      s += cabsf(vt[row + col * _N]);
    }
    s_upperbound = (s_upperbound > s) ? s_upperbound : s;
  }
  if (s_upperbound < lambda * lambda) {
    for (int bj = 0; bj < _N; bj++) {
#pragma simd
      for (int bi = 0; bi < _M; bi++) {
        buf[bi + bj * _M] = 0.;
      }
    }
    return;
  }
  cgesvd("S", "S", &_M, &_N, buf, &_M, s, u, &_M, vt, &_N, work, &lwork, rwork,
         &info);

  for (int bi = 0; bi < _N; bi++) {
    float sf = s[bi];
    for (int bj = 0; bj < _N; bj++) {
      vt[bi + bj * _N] *= (sf < lambda) ? 0.f : sf - lambda;
    }
  }

  cgemm("N", "N", &_M, &_N, &_N, &alpha, u, &_M, vt, &_N, &beta, buf, &_M);
}

void qrmysvthresh(complex float *buf, MKL_INT _M, MKL_INT _N, float *s,
                  complex float *u, complex float *vt, complex float *q,
                  complex float *tau, complex float *r, complex float *work,
                  MKL_INT lwork, float *rwork, MKL_INT info, float lambda) {
  complex float alpha = 1.0f;
  complex float beta = 0.0f;
  complex float zerocheck = 0.;
  for(int i = 0 ; i < _M*_N ; i++)
  {
    zerocheck += buf[i];
  }
  if (zerocheck == 0.) {
    for (int bj = 0; bj < _N; bj++) {
#pragma simd
      for (int bi = 0; bi < _M; bi++) {
        buf[bi + bj * _M] = 0.;
      }
    }
    return;
  }

  // 1. QR of A, and get R
  // Init R to zeroes
  for (int i = 0; i < _N; ++i)
    for (int j = 0; j < _N; ++j)
      r[i + j * _N] = .0;
  clacpy("N", &_M, &_N, buf, &_M, q, &_M);
  cgeqrf(&_M, &_N, q, &_M, tau, work, &lwork, &info);
  clacpy("U", &_M, &_N, q, &_M, r, &_N);

  // 2. Syrk to check for early stop
  csyrk("L", "T", &_N, &_N, &alpha, r, &_N, &beta, buf, &_N);
  float s_upperbound = 0;
  for (int i = 0; i < _N; i++) {
    float s = 0;
    for (int j = 0; j < _N; j++) {
      int row = (i > j ? i : j);
      int col = (i < j ? i : j);
      s += cabsf(buf[row + col * _N]);
    }
    s_upperbound = (s_upperbound > s) ? s_upperbound : s;
  }
  if (s_upperbound < lambda * lambda) {
    for (int bj = 0; bj < _N; bj++) {
#pragma simd
      for (int bi = 0; bi < _M; bi++) {
        buf[bi + bj * _M] = 0.;
      }
    }
    return;
  }

  // 3. SVD of R
  cgesvd("S", "S", &_N, &_N, r, &_N, s, u, &_N, vt, &_N, work, &lwork, rwork,
         &info);

  // 4. THR
  for (int bi = 0; bi < _N; bi++) {
    float sf = s[bi];
    for (int bj = 0; bj < _N; bj++) {
      vt[bi + bj * _N] *= (sf < lambda) ? 0.f : sf - lambda;
    }
  }

  // 5. GEMM with USV and last with Q
  for (int bj = 0; bj < _N; bj++) {
#pragma simd
    for (int bi = 0; bi < _M; bi++)
      buf[bi + bj * _M] = .0;
  }
  cgemm("N", "N", &_N, &_N, &_N, &alpha, u, &_N, vt, &_N, &beta, buf, &_M);
  cunmqr("L", "N", &_M, &_N, &_N, q, &_M, tau, buf, &_M, work, &lwork, &info);
}

void mylrthresh(const complex float *mat1, complex float *mat2, float lambda, int M,
                int N, int nimg, int nmap, int blksize, int shift0, int shift1) {
	printf("ths is only a test\n");
#pragma omp parallel
  {
    complex float *buf = (complex float *)malloc(blksize * blksize * nimg *
                                                 sizeof(complex float));
    MKL_INT _M = blksize * blksize;
    MKL_INT _N = nimg;
    complex float worksize;
    MKL_INT lwork = -1;
    MKL_INT info = 0;
    float *s = (float *)malloc(_N * sizeof(float));
    complex float *u = (complex float *)malloc(_M * _N * sizeof(complex float));
    complex float *vt =
        (complex float *)malloc(_N * _N * sizeof(complex float));
    complex float *u_qr =
        (complex float *)malloc(_N * _N * sizeof(complex float));
    complex float *q = (complex float *)malloc(_M * _N * sizeof(complex float));
    complex float *r = (complex float *)malloc(_N * _N * sizeof(complex float));
    complex float *tau = (complex float *)malloc(_N * sizeof(complex float));

    cgesvd("S", "S", &_M, &_N, buf, &_M, s, u, &_M, vt, &_N, &worksize, &lwork,
           NULL, &info);
    lwork = (MKL_INT)worksize;

    complex float *work =
        (complex float *)malloc(lwork * sizeof(complex float));
    float *rwork = (float *)malloc(_N * sizeof(float));

    int Mpad = blksize * ((M + blksize - 1) / blksize);
    int Npad = blksize * ((N + blksize - 1) / blksize);

    for (int m = 0; m < nmap; m++) {
#pragma omp for collapse(2)
      for (int i = 0 ; i < M; i += blksize) {
        for (int j = 0 ; j < N; j += blksize) {
	  int shiftedi = i - shift0;
	  int shiftedj = j - shift1;
          if ((shiftedi >= 0 ) && (shiftedj >=0 ) && (shiftedi + blksize <= M) && (shiftedj + blksize <= N)) {
            for (int img = 0; img < nimg; img++) {
              for (int bi = 0; bi < blksize; bi++) {
#pragma simd
                for (int bj = 0; bj < blksize; bj++) {
                  buf[bj + bi * blksize + img * blksize * blksize] = mat1
                      [shiftedj + bj + (shiftedi + bi) * N + m * M * N + img * nmap * M * N];
                }
              }
            }
          } else {
            for (int img = 0; img < nimg; img++) {
              for (int bi = 0; bi < blksize; bi++) {
                for (int bj = 0; bj < blksize; bj++) {
		  int bii = (shiftedi + bi);
		  if(bii < 0) bii = Mpad+bii;
		  bii = bii % M;
		  int bjj = (shiftedj + bj);
		  if(bjj < 0) bjj = Npad+bjj;
		  bjj = bjj % N;
                  buf[bj + bi * blksize + img * blksize * blksize] =
                      mat1[bjj + bii * N + m * M * N + img * nmap * M * N];
                }
              }
            }
          }
          mysvthresh(buf, _M, _N, s, u, vt, work, lwork, rwork, info,
           lambda);
          //qrmysvthresh(_buf, _M, _N, _s, _u_qr, _vt, _q, _tau, _r, _work, lwork, _rwork,
          //             info, _lambda);

          if ((shiftedi >= 0) && (shiftedj >= 0) && (shiftedi + blksize <= M) && (shiftedj + blksize <= N)) {
            for (int img = 0; img < nimg; img++) {
              for (int bi = 0; bi < blksize; bi++) {
#pragma simd
                for (int bj = 0; bj < blksize; bj++) {
                  mat2[shiftedj + bj + (shiftedi + bi) * N + m * M * N + img * nmap * M * N] =
                      buf[bj + bi * blksize + img * blksize * blksize];
                }
              }
            }
          } else {
            for (int img = 0; img < nimg; img++) {
              for (int bi = 0; bi < blksize; bi++) {
                for (int bj = 0; bj < blksize; bj++) {
		  int bii = (shiftedi + bi);
		  if(bii < 0) bii = Mpad+bii;
		  int bjj = (shiftedj + bj);
		  if(bjj < 0) bjj = Npad+bjj;
		  if((bii >= 0) && (bjj >= 0) && (bii < M) && (bjj < N))
		  {
                    mat2[bjj + (bii) * N + m * M * N +
                         img * nmap * M * N] =
                        buf[bj + bi * blksize + img * blksize * blksize];
		  }
                }
              }
            }
          }
        }
      }
    }

    free(buf);
    free(s);
    free(u);
    free(vt);
    free(work);
    free(rwork);
  }
}
