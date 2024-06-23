console.log("Hello from bart_worker")

// malloc code based on script from Christian Tönnes

function alloc(module, nbytes) {

    let ptr = module._malloc(nbytes);
    return new Uint8Array(module.HEAPU8.buffer, ptr, nbytes);
}

function allocFromString(module, string) {

    let heapArray = alloc(module, string.length + 1);

    heapArray[string.length] = 0;

    for (let i = 0; i < string.length; i++)
        heapArray[i] = string.charCodeAt(i);

    return heapArray;
}

function allocFromStringArray(module, inArgv) {

    let heapArgv = alloc(module, inArgv.length * 4);

    let heapArgv32 = new Int32Array(module.HEAPU8.buffer, heapArgv.byteOffset, inArgv.length);

    for (let k in inArgv) {

        let heapArray = allocFromString(module, inArgv[k]);

        let heapArray_byteOffset = heapArray.byteOffset;
        heapArgv32[k] = heapArray_byteOffset;
    }

    return heapArgv;
}

function allocFromArray(module, ar) {

    let heapArray = alloc(module, ar.length * ar.BYTES_PER_ELEMENT);

    heapArray.set(new Uint8Array(ar.buffer));
    return heapArray;
}

//////////////////////

var bart_module;
var stdout = "";
var stderr = "";

async function reload_bart(data) {
    bart_module = await bart_main({
        preRun : [],
        postRun: [],
        print: (function() {
            return function(text) {
                if (text)
                    stdout += text + "\n";
            };
        })(),
        printErr: function(text) {
            if (text)
                stderr += text + "\n";
        },
        setStatus: function(text) { },
        monitorRunDependencies: function(left) { },
        noInitialRun: true
    })

    console.log("Bart loaded");
    return null;
}

async function put_file(data)
{
    bart_module.FS.writeFile(data[1], data[2]);
    return null;
}

async function get_file(data)
{
    x = bart_module.FS.readFile(data[1])
    return x;
}

async function rm_file(data)
{
    bart_module.FS.unlink(data[1])
    return null;
}

var bart_state = 'idle';

async function bart_cmd(data) {

    bart_state = 'prep';
    argv = data[1].trim().split(/\s+/)

    stdout = "";
    stderr = "";

    let argv_heap = allocFromStringArray(bart_module, argv);
    let argv_heap_offset = argv_heap.byteOffset;
    let argc = argv.length;

    let ret = 255;
    let rt_error = null;

    try {
        var t = performance.now();
        bart_state = 'run';
        ret = bart_module.ccall("main", "number", ["number", "number"], [argc, argv_heap_offset])
        bart_state = 'done';
        console.log("Runtime ccall:", performance.now() - t, " ms");
    } catch(e) {
        // e is a bit weird, to be precise
        //  - it breaks console.log,
        //  - it breaks postMessage!
        // thus it's catched here.
        // furthermore, if exit/abort are called in the c code, it causes a runtime error
        // which would break bart tool -h.
        console.log("WASM Runtime error occured:", e.message);
        rt_error = e.message;
    }

    bart_module._free(argv_heap_offset);
    for(var k in argv_heap)
        bart_module._free(argv_heap[k]);

    return { 'ret': ret, 'stdout': stdout, 'stderr': stderr, 'rt_error': rt_error }
}



rpc_calls = {
    'reload_bart': reload_bart,
    'put_file': put_file,
    'get_file': get_file,
    'rm_file': rm_file,
    'bart_cmd': bart_cmd
}


var rpc_state = 'idle';

async function handle_msg(e) {

    rpc_state = 'msg received';
    data = e.data;

    console.log(`Bart webworker msg received: ${data[0]}`)

    if (!(data[0] in rpc_calls)) {

        console.log("Invalid msg type")
        postMessage([1, null]);
    }

    try {

        rpc_state = 'running';
        x = await rpc_calls[data[0]](data)
        rpc_state = 'success';
        postMessage([0, x]);
    } catch(e) {

        rpc_state = 'fail';
        console.log('rpc failed: ', data, e)
        postMessage([1, `rpc failed: ${data}, ${e}`, data, e]);
    }
}


// get emscripten assembly
importScripts('bart_main.js')

onmessage = handle_msg
