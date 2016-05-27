# Copyright 2013-2015. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.
#
# Authors: 
# 2013 Martin Uecker <uecker@eecs.berkeley.edu>
# 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>

import os  # Miscellaneous operating system interfaces
import numpy as np  # NumPy (multidimensional numerical arrays library)

# ======================================================================
def readcfl(
        basename,
        dirpath='.'):
    """
    Read CFL header+data pair

    Args:
        basename (str): the base filename. Corresponding '.hdr' and '.cfl' files
            must exist
        dirpath (str): the working directory

    Returns:
        array (ndarray): The CFL data
    """
    filepath = os.path.join(dirpath, basename)

    # load header
    with open(filepath + '.hdr', 'r') as header_file:
        header_file.readline()  # skip comment line
        dim_line = header_file.readline()

    # obtain the shape of the image
    shape = [int(i) for i in dim_line.split(' ') if int(i) > 1]
    data_size = int(np.prod(shape))

    # load data
    with open(filepath + ".cfl", "r") as data_file:
        array = np.fromfile(
            data_file, dtype=np.complex64, count=data_size)

    # note: BART uses FORTRAN-style memory allocation
    return array.reshape(shape, order='F')


# ======================================================================
def writecfl(
        array,
        basename,
        dirpath='.'):
    """
    Write CFL header+data pair

    Args:
        array (ndarray):
        basename (str): the base filename. Corresponding '.hdr' and '.cfl' files
            will be created/overwritten
        dirpath (str): the working directory

    Returns:
        None
    """
    filepath = os.path.join(dirpath, basename)

    # save header
    with open(filepath + '.hdr', 'w') as header_file:
        header_file.write(str('# Dimensions'))
        header_file.write(str(' '.join([str(n) for n in array.shape])))

    # save data
    with open(filepath + '.cfl', 'w') as data_file:
        # note: BART uses FORTRAN-style memory allocation
        array.astype(np.complex64, 'F').tofile(data_file)
