import sys
import traceback
import os

try:
    with open(sys.argv[1], 'r') as fd:
        for line in fd.readlines():
            exec(line)               
except:
    exc_info = sys.exc_info()
    traceback.print_exception(*exc_info)
    print('Exception occurred while executing line: ', line)
    sys.exit(1)

