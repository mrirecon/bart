# Copyright 2013-2015. The Regents of the University of California.
# Copyright 2021. Uecker Lab. University Center Göttingen.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2013 Martin Uecker <uecker@eecs.berkeley.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>

from __future__ import print_function
from __future__ import with_statement

import numpy as np
import mmap
import os


def readcfl(name):
    # get dims from .hdr
    with open(name + ".hdr", "rt") as h:
        h.readline() # skip
        l = h.readline()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    with open(name + ".cfl", "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n);
    return a.reshape(dims, order='F') # column-major

def readmulticfl(name):
    # get dims from .hdr
    with open(name + ".hdr", "rt") as h:
        lines = h.read().splitlines()

    index_dim = 1 + lines.index('# Dimensions')
    total_size = int(lines[index_dim])
    index_sizes = 1 + lines.index('# SizesDimensions')
    sizes = [int(i) for i in lines[index_sizes].split()]
    index_dims = 1 + lines.index('# MultiDimensions')

    with open(name + ".cfl", "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=total_size)

    offset = 0
    result = []
    for i in range(len(sizes)):
        dims = ([int(i) for i in lines[index_dims + i].split()])
        n = np.prod(dims)
        result.append(a[offset:offset+n].reshape(dims, order='F'))
        offset += n

    if total_size != offset:
        print("Error")

    return result


def writecfl(name, array):
    with open(name + ".hdr", "wt") as h:
        h.write('# Dimensions\n')
        for i in (array.shape):
                h.write("%d " % i)
        h.write('\n')

    size = np.prod(array.shape) * np.dtype(np.complex64).itemsize

    with open(name + ".cfl", "a+b") as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        if array.dtype != np.complex64:
            array = array.astype(np.complex64)
        mm.write(np.ascontiguousarray(array.T))
        mm.close()
        #with mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE) as mm:
        #    mm.write(array.astype(np.complex64).tobytes(order='F'))

def writemulticfl(name, arrays):
    size = 0
    dims = []

    for array in arrays:
        size += array.size
        dims.append(array.shape)

    with open(name + ".hdr", "wt") as h:
        h.write('# Dimensions\n')
        h.write("%d\n" % size)

        h.write('# SizesDimensions\n')
        for dim in dims:
            h.write("%d " % len(dim))
        h.write('\n')

        h.write('# MultiDimensions\n')
        for dim in dims:
            for i in dim:
                h.write("%d " % i)
            h.write('\n')
            
    size = size * np.dtype(np.complex64).itemsize

    with open(name + ".cfl", "a+b") as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        for array in arrays:
            if array.dtype != np.complex64:
                array = array.astype(np.complex64)
            mm.write(np.ascontiguousarray(array.T))
        mm.close()
