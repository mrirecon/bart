import argparse
import numpy as np
import h5py

from bart import cfl

def convert(infile, outfile):
    x = h5py.File(infile, 'r')
    dat = np.array(x['dataset']['data'])['data']

    cmplx = np.zeros((len(dat[0])//2, len(dat)), dtype=np.complex64)

    for i,acq in enumerate(dat):
        cmplx[:,i].real = acq[::2]
        cmplx[:,i].imag = acq[1::2]

    cfl.writecfl(outfile, cmplx)


parser = argparse.ArgumentParser()
parser.add_argument("infile")
parser.add_argument("outfile")
args = parser.parse_args()

convert(args.infile, args.outfile)
