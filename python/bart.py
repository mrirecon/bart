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

def bart(nargout, cmd, *args, **kwargs):

    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguments...>)")
        return

    try:
        bart_path = os.environ['TOOLBOX_PATH']
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

    for idx in range(nargin):
        cfl.writecfl(infiles[idx], args[idx])

    args_kw = ["--" if len(kw)>1 else "-" + kw for kw in kwargs]
    infiles_kw = [name + 'in' + kw for kw in kwargs]
    for idx, kw in enumerate(kwargs):
        cfl.writecfl(infiles_kw[idx], kwargs[kw])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]

    cmd = cmd.split(" ")

    if os.name =='nt':
        if isWSL:
            #For WSL and modify paths
            infiles = [PathCorrection(item) for item in infiles]
            infiles_kw = [PathCorrection(item) for item in infiles_kw]
            outfiles = [PathCorrection(item) for item in outfiles]
            cmd = [PathCorrection(item) for item in cmd]
            args_infiles_kw = [item for pair in zip(args_kw, infiles_kw) for item in pair]
            shell_cmd = ['wsl', 'bart', *cmd, *args_infiles_kw, *infiles, *outfiles]
        else:
            #For cygwin use bash and modify paths
            infiles = [item.replace(os.path.sep, '/') for item in infiles]
            infiles_kw = [item.replace(os.path.sep, '/') for item in infiles_kw]
            outfiles = [item.replace(os.path.sep, '/') for item in outfiles]
            cmd = [item.replace(os.path.sep, '/') for item in cmd]
            args_infiles_kw = [item for pair in zip(args_kw, infiles_kw) for item in pair]
            shell_cmd = ['bash.exe', '--login',  '-c', os.path.join(bart_path, 'bart'), *cmd, *args_infiles_kw, *infiles, *outfiles]
            #TODO: Test with cygwin, this is just translation from matlab code
    else:
        args_infiles_kw = [item for pair in zip(args_kw, infiles_kw) for item in pair]
        shell_cmd = [os.path.join(bart_path, 'bart'), *cmd, *args_infiles_kw, *infiles, *outfiles]

    # run bart command
    ERR, stdout, stderr = execute_cmd(shell_cmd)

    # store error code, stdout and stderr in function attributes for outside access
    # this makes it possible to access these variables from outside the function (e.g "print(bart.ERR)")
    bart.ERR, bart.stdout, bart.stderr = ERR, stdout, stderr

    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    for elm in infiles_kw:
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
        print(f"Command exited with error code {ERR}.")
        return

    if nargout == 0:
        return
    elif nargout == 1:
        return output[0]
    else:
        return output


def execute_cmd(cmd):
    """
    Execute a command in a shell.
    Print and catch the output.
    """
    
    errcode = 0
    stdout = ""
    stderr = ""

    # remove empty strings from cmd
    cmd = [item for item in cmd if len(item)]

    # execute cmd
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)

    # print to stdout
    for stdout_line in iter(proc.stdout.readline, ""):
        stdout += stdout_line
        print(stdout_line, end="")
    proc.stdout.close()

    # in case of error, print to stderr
    errcode = proc.wait()
    if errcode:
        stderr = "".join(proc.stderr.readlines())
        print(stderr)
    proc.stderr.close()

    return errcode, stdout, stderr
