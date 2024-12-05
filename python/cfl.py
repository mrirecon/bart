# Copyright 2013-2015. The Regents of the University of California.
# Copyright 2021. Uecker Lab. University Center Göttingen.
# Copyright 2024. Institute for Biomedical Imaging. TU Graz.
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

# see src/misc/io.c for rawarray header definition
_RA_MAGIC = int(0x7961727261776172)
_RA_TYPE_COMPLEX = int(4)
_RA_CFL_SIZE = int(8)
_RA_HEADER_ELEMS = 6

def _readra(name):

    with open(name, "rb") as f:
        header = np.fromfile(f, dtype=np.uint64, count=_RA_HEADER_ELEMS)
        magic = header[0]
        flags = header[1]
        eltype = header[2]
        elsize = header[3]
        datasize = header[4]
        ndims = header[5]

        if ( magic != _RA_MAGIC
                or (flags & np.uint64(1)) != 0
                or eltype != _RA_TYPE_COMPLEX
                or elsize != _RA_CFL_SIZE ):
            print("Invalid .ra header!")
            raise RuntimeError

        shape_arr = np.fromfile(f, dtype=np.uint64, count = ndims)

        arr = np.fromfile(f, dtype=np.complex64, count = datasize // elsize).reshape(shape_arr, order='F')
    return arr

def _writera(name, array):

    header = np.empty((6,), dtype=np.uint64)
    header[0] = _RA_MAGIC
    header[1] = np.uint64(0)
    header[2] = _RA_TYPE_COMPLEX
    header[3] = _RA_CFL_SIZE
    header[4] = np.prod(array.shape) * np.dtype(np.complex64).itemsize
    header[5] = array.ndim


    shape_arr = np.array(array.shape, dtype=np.uint64)
    fullsize = int(header[4] + header.nbytes + shape_arr.nbytes)

    with open(name, "a+b") as d:
        os.ftruncate(d.fileno(), fullsize)
        mm = mmap.mmap(d.fileno(), fullsize, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        if array.dtype != np.complex64:
            array = array.astype(np.complex64)
        mm.write(np.ascontiguousarray(header))
        mm.write(np.ascontiguousarray(shape_arr))
        mm.write(np.ascontiguousarray(array.T))
        mm.close()

    return


def readcfl(name):

    if name.endswith(".ra"):
        return _readra(name)

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

    if name.endswith(".ra"):
        return _writera(name, array)

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
