# Copyright 2016. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by 
# a BSD-style license which can be found in the LICENSE file.
#
# Authors: 
# 2016 Siddharth Iyer <sid8795@gmail.com>
# 2018 Soumick Chatterjee <soumick.chatterjee@ovgu.de> , WSL Support

import subprocess as sp
import tempfile as tmp
import os
import sys

if __spec__.parent:
    from . import cfl
    from .wslsupport import PathCorrection
else:
    # 'traditional' copy-paste bart.py
    from wslsupport import PathCorrection
    import cfl

isWASM = True if sys.platform == 'emscripten' else False

def bart(nargout, cmd, *args, **kwargs):
    if isWASM:
        print("Please use await bart.bart2!", file=sys.stderr)
        raise RuntimeError("Synchronous bart not available in wasm")

    prep = bart_prepare(nargout, cmd, *args, **kwargs)
    if not prep:
        return

    ERR, stdout, stderr = execute_cmd(prep['shell_cmd'])

    # store error code, stdout and stderr in function attributes for outside access
    # this makes it possible to access these variables from outside the function (e.g "print(bart.ERR)")
    bart.ERR, bart.stdout, bart.stderr = ERR, stdout, stderr

    return bart_postprocess(nargout, ERR, prep['infiles'], prep['infiles_kw'], prep['outfiles'])


async def bart2(nargout, cmd, *args, **kwargs):
    if not isWASM:
        print("Please use synchronous bart.bart!", file=sys.stderr)
        raise RuntimeError("Asynchronous bart is only available in wasm")

    prep = bart_prepare(nargout, cmd, *args, **kwargs)
    if not prep:
        return

    ERR, stdout, stderr = await run_wasm_cmd(**prep)

    # store error code, stdout and stderr in function attributes for outside access
    # this makes it possible to access these variables from outside the function (e.g "print(bart.ERR)")
    bart.ERR, bart.stdout, bart.stderr = ERR, stdout, stderr

    return bart_postprocess(nargout, ERR, prep['infiles'], prep['infiles_kw'], prep['outfiles'])


def bart_prepare(nargout, cmd, *args, **kwargs):
    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguments...>)")
        return

    try:
        bart_path = os.environ['BART_TOOLBOX_PATH']
    except:
        bart_path = None
    # support old environment variable:
    if bart_path is None:
        try:
            bart_path = os.environ['TOOLBOX_PATH']
        except:
            bart_path = None

    isWSL = False

    if not isWASM and not bart_path:
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
                raise Exception('Environment variable BART_TOOLBOX_PATH is not set.')

    name = tmp.NamedTemporaryFile().name

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]

    for idx in range(nargin):
        cfl.writecfl(infiles[idx], args[idx])

    args_kw = [("--" if len(kw)>1 else "-") + kw for kw in kwargs]
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

        assert(not isWASM)

    else:
        args_infiles_kw = [item for pair in zip(args_kw, infiles_kw) for item in pair]
        shell_cmd = [os.path.join(bart_path, 'bart') if not isWASM else 'bart', *cmd, *args_infiles_kw, *infiles, *outfiles]

    return dict(shell_cmd=shell_cmd, infiles=infiles, infiles_kw=infiles_kw, outfiles=outfiles)


def bart_postprocess(nargout, ERR, infiles, infiles_kw, outfiles):
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



wasm_bart_ok = False;

async def get_wasm_cfl(name):
    await wasm_async_call(f"get_cfl('{name}')")

async def put_wasm_cfl(name):
    await wasm_async_call(f"send_cfl('{name}')")

async def rm_bart_cfl(name):
    await wasm_async_call(f"rm_cfl('{name}')")

async def wasm_load_bart():
    global wasm_bart_ok
    await wasm_async_call("reload_bart()")
    wasm_bart_ok = True

async def run_wasm_cmd(shell_cmd, infiles, infiles_kw, outfiles):
    global wasm_bart_ok;
    try:
        if not wasm_bart_ok:
            await wasm_load_bart()

        for f in infiles + infiles_kw:
            await put_wasm_cfl(f)

        non_empty_cmd = [x for x in shell_cmd if len(shell_cmd) > 0]

        result = await wasm_async_call("bart_cmd('" + ' '.join(non_empty_cmd) + "')")
        ERR, stdout, stderr = result['ret'], result['stdout'], result['stderr']

        if not stdout is None and len(stdout.strip()) > 0:
            print(stdout)

        if not stderr is None and len(stderr.strip()) > 0:
            print(stderr, file=sys.stderr)

        if not 0 == ERR:
            print(f"Function exited with {ERR}", file=sys.stderr)

        for f in outfiles:
            await get_wasm_cfl(f)

        for f in infiles + infiles_kw + outfiles:
            await rm_bart_cfl(f)

        return ERR, stdout, stderr

    except Exception as e:
        print("Exception in bart worker calls occurred:")
        print(e)
        wasm_bart_ok = False
        raise e

async def wasm_async_call(cmd):
    # synchronous function would be nice
    # but this seems impossible: https://github.com/pyodide/pyodide/issues/3932
    #loop = asyncio.get_event_loop()
    #task = pyodide.code.run_js(cmd)
    #return loop.run_until_complete(asyncio.wait([task]))

    ret = (await pyodide.code.run_js(cmd)).to_py()
    if 0 != ret[0]:
        raise Exception(f"Error in JS call: {ret[1]}")

    return ret[1];


if isWASM:
    import pyodide, pyodide_js, js

    # Export pyodide to webworker namespace:
    pyodide.code.run_js("var pyodide;")
    js.pyodide = pyodide_js

    # load BART:
    pyodide.code.run_js("""
                        if ('undefined' == (typeof window)) {
                            importScripts("bart_base.js");

                        } else {

                            script = document.createElement('script');
                            script.type = 'text/javascript';
                            script.async = true;
                            script.src = "bart_base.js";
                            document.body.appendChild(script);
                        }
                        """)
