# Copyright 2013-2015. The Regents of the University of California.
# Copyright 2021. Uecker Lab. University Center Göttingen.
# Copyright 2024. Institute for Biomedical Imaging. TU Graz.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2013 Martin Uecker <uecker@eecs.berkeley.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>


import numpy as np

# see src/misc/io.c for rawarray header definition
_RA_MAGIC = np.uint64(0x7961727261776172)
_RA_TYPE_COMPLEX = np.uint64(4)
_RA_CFL_SIZE = np.uint64(8)
_RA_FLAG_BIG_ENDIAN = np.uint64(1)
_RA_HEADER_ELEMS = 6


def _readra(name):
    with open(name, "rb") as f:
        magic, flags, eltype, elsize, datasize, ndims = np.fromfile(
            f, dtype=np.uint64, count=_RA_HEADER_ELEMS)

        if (magic != _RA_MAGIC
                or flags & _RA_FLAG_BIG_ENDIAN
                or eltype != _RA_TYPE_COMPLEX
                or elsize != _RA_CFL_SIZE):
            raise RuntimeError(f"Invalid .ra header: {name}")

        dims = np.fromfile(f, dtype=np.uint64, count=ndims)

        # remove trailing singleton dimensions
        while dims.size > 1 and dims[-1] == 1:
            dims = dims[:-1]

        return np.fromfile(f, dtype=np.complex64, count=int(datasize // 8)
                           ).reshape(dims, order='F')


def _writera(name, array):

    if array.dtype != np.complex64:
        array = array.astype(np.complex64)

    header = np.array([
        _RA_MAGIC,
        0,
        _RA_TYPE_COMPLEX,
        _RA_CFL_SIZE,
        array.nbytes,
        array.ndim
    ], dtype=np.uint64)
    shape_arr = np.array(array.shape, dtype=np.uint64)

    with open(name, "wb") as f:
        f.write(header)
        f.write(shape_arr)
        f.write(np.ascontiguousarray(array.T))


def readcfl(name):

    if name.endswith(".ra"):
        return _readra(name)

    # get dims from .hdr
    with open(name + ".hdr", "rt") as f:
        next(f)  # skip first line
        line = next(f)
    dims = [int(i) for i in line.split()]

    # remove singleton dimensions from the end
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()

    # load data and reshape into dims
    with open(name + ".cfl", "rb") as f:
        return np.fromfile(f, dtype=np.complex64, count=np.prod(dims)
                           ).reshape(dims, order='F')  # column-major


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

    with open(name + ".hdr", "wt") as f:
        f.write('# Dimensions\n')
        f.write(" ".join(str(i) for i in array.shape))
        f.write('\n')

    if array.dtype != np.complex64:
        array = array.astype(np.complex64)

    with open(name + ".cfl", "wb") as f:
        f.write(np.ascontiguousarray(array.T))


def writemulticfl(name, arrays):
    total_size = sum(arr.size for arr in arrays)
    dims = [arr.shape for arr in arrays]

    with open(name + ".hdr", "wt") as f:
        f.write('# Dimensions\n')
        f.write("%d\n" % total_size)
        f.write('# SizesDimensions\n')
        f.write(' '.join(str(len(dim)) for dim in dims))
        f.write('\n')
        f.write('# MultiDimensions\n')
        for dim in dims:
            f.write(' '.join(str(i) for i in dim))
            f.write('\n')

    with open(name + ".cfl", "wb") as f:
        for array in arrays:
            if array.dtype != np.complex64:
                array = array.astype(np.complex64)
            f.write(np.ascontiguousarray(array.T))
