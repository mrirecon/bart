# Copyright 2013. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.
#
# Authors: 
# 2013 Martin Uecker <uecker@eecs.berkeley.edu>


import operator;
import numpy;

def readcfl(name):
	h = open(name + ".hdr", "r")
	h.readline() # skip
	l = h.readline()
	dims = [int(i) for i in l.split( )]
	n = reduce(operator.mul, dims, 1)
	h.close()
	d = open(name + ".cfl", "r")
	a = numpy.fromfile(d, dtype=numpy.complex64, count=n);
	d.close()
	return a.reshape(dims[::-1]);

	
def writecfl(name, array):
	h = open(name + ".hdr", "w")
	h.write('# Dimensions\n')
	for i in reversed(array.shape):
		h.write("%d " % i)
	h.write('\n')
	h.close()
	d = open(name + ".cfl", "w")
	array.astype(numpy.complex64).tofile(d)
	d.close()


