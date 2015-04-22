# Copyright 2013-2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.
#
# Authors: 
# 2013 Martin Uecker <uecker@eecs.berkeley.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>


import numpy as np

def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n);
    d.close()
    return a.reshape(dims, order='F') # column-major

	
def writecfl(name, array):
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()
