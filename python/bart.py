# Copyright 2016. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.
#
# Authors: 
# 2016 Siddharth Iyer <sid8795@gmail.com>
# 2018 Soumick Chatterjee <soumick.chatterjee@ovgu.de> , WSL Support

import subprocess as sp
import tempfile as tmp
import cfl
import os
from wslsupport import PathCorrection

def bart(nargout, cmd, *args):

    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguements...>)")
        return None

    try:
        bart_path = os.environ['TOOLBOX_PATH'] + '/bart '
    except:
        bart_path = None
    isWSL = False

    if not bart_path:
        if os.path.isfile('/usr/local/bin/bart'):
            bart_path = '/usr/local/bin'
        elif os.path.isfile('/usr/bin/bart'):
            bart_path = '/usr/bin'
        else:
            bartstatus = os.system('wsl bart version -V')
            if bartstatus==0:
                bart_path = '/usr/bin'
                isWSL = True
            else:
                raise Exception('Environment variable TOOLBOX_PATH is not set.')

    name = tmp.NamedTemporaryFile().name

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]
    in_str = ' '.join(infiles)

    for idx in range(nargin):
        cfl.writecfl(infiles[idx], args[idx])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]
    out_str = ' '.join(outfiles)

    if os.name =='nt':
        if isWSL:
            #For WSL and modify paths
            cmdWSL = PathCorrection(cmd)
            in_strWSL = PathCorrection(in_str)
            out_strWSL =  PathCorrection(out_str)	
            ERR = os.system('wsl bart ' + cmdWSL + ' ' + in_strWSL + ' ' + out_strWSL)
        else:
            #For cygwin use bash and modify paths
            ERR = os.system('bash.exe --login -c ' + bart_path + '"/bart ' + cmd.replace(os.path.sep, '/') + ' ' + in_str.replace(os.path.sep, '/') + ' ' + out_str.replace(os.path.sep, '/') + '"')
            #TODO: Test with cygwin, this is just translation from matlab code
    else:
        ERR = os.system(bart_path + '/bart ' + cmd + ' ' + in_str + ' ' + out_str)

    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    output = []
    for idx in range(nargout):
        elm = outfiles[idx]
        if not ERR:
            output.append(cfl.readcfl(elm))
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    if ERR:
        raise Exception("Command exited with an error.")

    if nargout == 1:
        output = output[0]

    return output
