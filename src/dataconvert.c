//
//  dataconvert.c
//
//
//  Author:
//  2015 Mariya Doneva
//
//

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"
#include "linops/someops.h"
#include "linops/linop.h"
#include "num/iovec.h"


#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"


//#include "dataconvert.h"


static void usage(const char* name, FILE* fd)
{
    fprintf(fd, "Usage: %s [options] <input> <output>\n", name);
}

static void help(void)
{
    printf( "\n"
           "Converting between k-space data exported from Philips reconstruction \n"
           "and BART raw data format \n"
           "\n"
           );
}


int main_dataconvert(int argc, char* argv[])
{
    // -----------------------------------------------------------
    // set up conf and option parser

    
    int c;
    while (-1 != (c = getopt(argc, argv, "fih"))) {
        switch(c) {
                
                
            case 'h':
                usage(argv[0], stdout);
                help();
                exit(0);
                
            default:
                usage(argv[0], stderr);
                exit(1);
        }
    }
    if (argc - optind != 2) {
        
        usage(argv[0], stderr);
        exit(1);
    }
    
    
    
    char filename[1024];
    int sizex, sizey, sizez, ncoils;
    int N = DIMS;
    long ksp_dims[N];
    FILE* fid;
    
    // -----------------------------------------------------------
    // load data
    //
    // read header file
    
    
    
    sprintf(filename, "%s.hdr",argv[optind + 0]);
    printf("header file name %s \n", filename);
    
    
    int value[N];
    char junk[10];
    fid = fopen(filename,"r");
    int p = 0;
    
    for (int i=0; i<4; i++)
    {
        p = fscanf(fid, "%s %d",junk, &value[i]);
        
        if (p!=EOF)
        {
            printf(" %s %d \n", junk, value[i]);
        }
        
    }
    
    fclose(fid);
    
    sizex = value[2];
    sizey = value[0];
    sizez = value[1];
    ncoils = value[3];
    
    
    
    // set dimensions of BART k-space data
    
    ksp_dims[0] = (long)sizex;
    ksp_dims[1] = (long)sizey;
    ksp_dims[2] = (long)sizez;
    ksp_dims[3] = (long)ncoils;
    
    
    for (int i = 4; i<N; i++) {
        ksp_dims[i] = (long)1;
    }
    
    
    
    // -----------------------------------------------------------
    // create output k-space data
    
    
    num_init();
    
    // complex float* kspace = create_cfl("test_output", DIMS, ksp_dims);
   
    complex float *kdata;
    complex float *kdata1;
    complex float *sos;
    
    kdata = create_cfl(argv[optind + 1], DIMS, ksp_dims);
    md_clear(DIMS, ksp_dims, kdata, CFL_SIZE);
    
    kdata1 = (complex float*)malloc(sizex*sizey*sizez*ncoils*sizeof(complex float));
    sos    = (complex float*)malloc(sizey*sizez*sizeof(complex float));

    
    sprintf(filename, "%s.dat",argv[optind + 0]);
    printf("input file name %s \n", filename);
    
    
    fid = fopen(filename,"rb");
    
    if(fid == NULL)
    {
        fprintf(stderr, "Can't open input file file_kdata.dat\n");
        exit(1);
    }
    else
    {
        fread(kdata1,sizeof(complex float),sizex*sizey*sizez*ncoils,fid);
        fclose(fid);
        printf("reading file done \n");
    }
    
    
   // printf("real part %f \n",creal(kdata1[0]));
   // printf("imaginary part %f \n",cimag(kdata1[0]));
    
    // reorder the data to kx, ky, kz, coils
    
    for (int i = 0; i<ncoils; i++) {
        for (int j=0; j<sizex; j++) {
            for (int k=0; k<sizey; k++) {
                for (int l=0; l<sizez; l++) {
                    kdata[i*sizex*sizey*sizez + l*sizex*sizey + k*sizex + j] = kdata1[j*sizey*sizez*ncoils + i*sizey*sizez + l*sizey + k];
                }
            }
        }
    }
    
    
    
    // adjust fftshifts and apply FFTs to get the corresponding FFT definition with shifts before and after
    // -----------------------------------------------------------
    // In the Philips recon the transform from kspace to image space is forward FFT, data are premodulated so
    // only one fft shift is required. The exported data is Fourier transformed along the readout direction
    // This means that the transform to BART raw data format requires 2 non-centered 2D ffts along the y and z directions, +1/-1 modulation in all three directions and finally one 1D fft along the readout direction
    
    
    // The two consecutive ffts can be replaced by flipping the data in y and z and possibly applying a circular shift in both directions. Perform consistency check by comparing the image phase of single coil image
    
    // Transform to image space
    
    fft(N,ksp_dims,6,kdata,kdata);
    md_zsmul(N, ksp_dims, kdata, kdata, 1./sqrt(sizey*sizez));
   
    // Apply a Fourier transform in yz
    
    fft(N,ksp_dims,6,kdata,kdata);
    md_zsmul(N, ksp_dims, kdata, kdata, 1./sqrt(sizey*sizez));
   
    // Apply +1/-1 modulation for centered FFT
  
    fftmod(N, ksp_dims, 7, kdata, kdata);
    
    // Apply a Fourier transform in the readout direction
    
    fft(N,ksp_dims,1,kdata,kdata);
    md_zsmul(N, ksp_dims, kdata, kdata, 1./sqrt(sizex));
    
    
    // Finally, the values k-space positions that have not been measured are set to 0
    
    md_zrss(N, ksp_dims, 9, sos, kdata);
    
    float max = 0.0;
    
    for (int i = 0; i<sizey*sizez; i++)
    {
        if (max < creal(sos[i]))
        {
            max = creal(sos[i]);
        }
    }
    
    
    float thresh = 0.0001*max;
   // printf("threshold = %f \n", max);

    
    
    for (int i = 0; i<sizey; i++) {
        for (int j =0; j<sizez; j++) {
            
            if (creal(sos[j*sizey+i])<thresh)
            {
                
                for (int k =0; k< sizex; k++) {
                    for (int l = 0; l<ncoils; l++) {
                        kdata[l*sizex*sizey*sizez + j*sizex*sizey + i*sizex + k] = (complex float) 0.0;
                    }
                }
                
            }
            
        }
    }
    
    
    
    // write cfl file for BART reconstruction
    
    unmap_cfl(N, ksp_dims, kdata);
    
    // cleanup
    
    free(kdata1);
    
    
    
    exit(0);
}