//
//  imageconvert.c
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
#include "num/iovec.h"


#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"



static void usage(const char* name, FILE* fd)
{
    fprintf(fd, "Usage: %s [options] <input> <output>\n", name);
}

static void help(void)
{
    printf( "\n"
           "Converting between image data reconstructed with BART"
           "and image data to plug in the Philips reconstruction \n"
           "and BART raw data format \n"
           "\n"
           );
}


int main_imageconvert(int argc, char* argv[])
{
    
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
    int sizex, sizey, sizez;
    int N = DIMS;
    long img_dims[N];
    long img_dims1[N];
    FILE* fid;
    
    // -----------------------------------------------------------
    // load image from cfl file
    //
    
    complex float *image;
    complex float *image1;
    
    image = load_cfl(argv[optind + 0], N, img_dims);
    
    num_init();

    // set dimensions for the output file
    // multiple output images are not considered at the moment reducing the dimensions to 3 spacial dimensions only
    
    img_dims1[0] = img_dims[0];
    img_dims1[1] = img_dims[1];
    img_dims1[2] = img_dims[2];
    
    for (int i = 3; i<N; i++) {
        img_dims1[i] = (long) 1;
    }
    
    
    sizex = img_dims[0];
    sizey = img_dims[1];
    sizez = img_dims[2];
    
    
    
    // -----------------------------------------------------------
    // create output image data
    
    image1 = md_alloc(N,img_dims1,sizeof(complex float));
    
    
    // perform fftshift in readout
    
    fftshift(N, img_dims, 1, image, image);

    
    // reorder data as y,z,x
    
    
    for (int i = 0; i<sizex; i++) {
        for (int j=0; j<sizey; j++) {
            for (int k=0; k<sizez; k++) {
                
                image1[i*sizey*sizez + k*sizey + j] = image[k*sizex*sizey + j*sizex + i];
                
            }
        }
    }
    
    
    // write output file
    
    
    sprintf(filename, "%s.dat",argv[optind + 1]);
    printf("output file name %s \n", filename);
    
    
    fid = fopen(filename,"wb");
    
    if(fid == NULL)
    {
        fprintf(stderr, "Can't open input file for writing \n");
        exit(1);
    }
    else
    {
        fwrite(image1,sizeof(float),2*sizex*sizey*sizez,fid);
        fclose(fid);
        printf("writing file done \n");
    }
    
    
    
    // cleanup
    
    md_free(image1);
    unmap_cfl(N,img_dims,image);
    
    
    exit(0);
}