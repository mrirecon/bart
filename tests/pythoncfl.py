#!/usr/bin/env python

from __future__ import print_function
import cfl
import sys

# from http://stackoverflow.com/questions/5574702/how-to-print-to-stderr-in-python
def errprint(*args, **kwargs):
	    print(*args, file=sys.stderr, **kwargs)

def main(out_name, in_name):
	input = cfl.readcfl(in_name)
	#cfl.writecfl(input, out_name)
	cfl.writecfl(out_name, input)
	return 0

if __name__ == '__main__':

	if len(sys.argv) != 3:
		errprint('Usage:', sys.argv[0], '<input> <output>')

	exit(main(sys.argv[2], sys.argv[1]))
