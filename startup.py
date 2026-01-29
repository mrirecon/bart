import os
import sys

if 'BART_TOOLBOX_PATH' in os.environ and os.path.exists(os.environ['BART_TOOLBOX_PATH']):
	sys.path.append(os.path.join(os.environ['BART_TOOLBOX_PATH'], 'python'))
else:
	raise RuntimeError("BART_TOOLBOX_PATH is not set correctly!")


from bart import bart
import cfl
