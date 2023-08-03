/* Copyright 2023. Christian Tönnes.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2023 Christian Tönnes <christian.toennes@keks.li>
 */

const DIMS = 16;
const scalar_size = 4;

function enter_cmd(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("btn_cmd").click();
    }
}

function cmd() {
    var cmds = document.getElementById("cmd_input").value.split(/\s*;\s*/);
    for(var cmd in cmds) {
        var argv = cmds[cmd].split(/\s+/);
        console.log(argv);
        var t = performance.now();
        var inArgv = allocFromStringArray(argv);
        var inArgv_byteOffset = inArgv.byteOffset;
        
        if(this["_main_"+argv[0]] == undefined) { 
            console.log("function:", "_main_"+argv[0], "was not exported.");
        }
        this["_main_"+argv[0]](argv.length, inArgv_byteOffset);

        _free(inArgv_byteOffset);
        for(var k in inArgv){
            _free(inArgv[k]);
        }
        console.log("Runtime", cmds[cmd], performance.now()-t);
    }
    list_memcfl();
}

var data = undefined;
var dims = undefined;

function display(file) {
    console.log(file);
    [data, dims] = from_memcfl(file);
    var dimx = document.getElementById("dimx");
    var dimy = document.getElementById("dimy");
    var dimz = document.getElementById("dimz");
    while(dimx.firstChild) {dimx.removeChild(dimx.lastChild);}
    while(dimy.firstChild) {dimy.removeChild(dimy.lastChild);}
    while(dimz.firstChild) {dimz.removeChild(dimz.lastChild);}
    for(var k in dims) {
        var opt = document.createElement("option");
        opt.value = k;
        opt.textContent = "Dim " + k + " ( " + dims[k] + " )";
        dimx.appendChild(opt);
        opt = document.createElement("option");
        opt.value = k;
        opt.textContent = "Dim " + k + " ( " + dims[k] + " )";
        dimy.appendChild(opt);
        opt = document.createElement("option");
        opt.value = k;
        opt.textContent = "Dim " + k + " ( " + dims[k] + " )";
        dimz.appendChild(opt);
        var idx = document.getElementById("idx_"+k).firstChild;
        idx.max = dims[k]-1;
        idx.min = 0;
        idx.value = 0;
        idx.onchange = function() {updateDisplay()};
        var dim = document.getElementById("dim_"+k);
        dim.textContent = ""+dims[k];
    }
    dimy.value = 1;
    dimz.value = 2;
    updateDisplay();
}

function calc_idx(idx) {
    var mult = 1;
    var index = 0;
    for(var i=0;i<16;i++) {
        if(i>0) {mult *= dims[i-1];}
        index += idx[i]*mult;
    }
    return index;
}

function updateDisplay() {
    var dimx = document.getElementById("dimx").value;
    var dimy = document.getElementById("dimy").value;
    
    var idx = [];
    for(var i=0;i<16;i++) {
        idx.push(document.getElementById("idx_"+i).firstChild.valueAsNumber);
    }    
    var canvas = document.getElementById("img");
    ctx = canvas.getContext('2d');
    canvas.width = dims[dimx];
    canvas.height = dims[dimy];
    idata = ctx.createImageData(dims[dimx], dims[dimy]);
    var result = new Uint8ClampedArray(dims[dimx] * dims[dimy] * 4);
    for(var x=0;x<dims[dimx];x++) {
        for(var y=0;y<dims[dimy];y++) {
            idx[dimx] = x;
            idx[dimy] = y;
            var j = calc_idx(idx);
            var i = y + x*dims[dimx];
            result[4*i] = data[2*j]*256;
            result[4*i+1] = data[2*j]*256;
            result[4*i+2] = data[2*j]*256;
            result[4*i+3] = 255;
        }
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);
}

function scroll(event) {
    console.log(event);
    var dimz = document.getElementById("dimz").value;
    var idx = document.getElementById("idx_"+dimz);
    if(event.deltaY>0) {
    if(idx.valueAsNumber>=dims[dimz]-1) { idx.value = dims[dimz]-1; }
    else { updateDisplay(); }
    } else {
    if(idx.valueAsNumber<=0) { idx.value = 0; }
    else { updateDisplay(); }
    }
}

function wrap(file, fun) {
    return function() {fun(file);};
}

function display_memcfl_files(files) {
    var list = document.getElementById("files");
    while(list.firstChild) {list.removeChild(list.lastChild);}
    for(var k in files) {
        var file = files[k];
        var li = document.createElement("li");
        li.appendChild(document.createTextNode(file));
        var rm = document.createElement("button");
        rm.onclick=wrap(file, unlink_memcfl);
        rm.textContent="unlink";
        li.appendChild(rm);
        var sh = document.createElement("button");
        sh.onclick=wrap(file, display);
        sh.textContent="display";
        li.appendChild(sh);
        list.appendChild(li);
    }
}

function to_memcfl(name, dims, data) {
    var heapDims = allocFromArray(dims);
    var heapDims_byteOffset = heapDims.byteOffset;
    var heapName = allocFromString(name);
    var heapName_byteOffset = heapName.byteOffset
    var memcfl_byteoffset = _memcfl_create(heapName_byteOffset, dims.length, heapDims_byteOffset);
    var memcfl = new Float32Array(Module.HEAPU8.buffer, memcfl_byteoffset, data.length);
    memcfl.set(data);
    return name;
}

function from_memcfl(name) {
    var heapDims = alloc(DIMS*scalar_size);
    var heapDims_byteOffset = heapDims.byteOffset;
    var heapName = allocFromString(name);
    var heapName_byteOffset = heapName.byteOffset
    var out_data = _load_cfl(heapName_byteOffset, DIMS, heapDims_byteOffset);
    
    var dims = Int32Array.from(new Int32Array(Module.HEAPU8.buffer, heapDims_byteOffset, DIMS));
    var size = 2;
    for(var dim in dims) {
        size = size*dims[dim];
    }
    var data = Float32Array.from(new Float32Array(Module.HEAPU8.buffer, out_data, size));
    return [data, dims];
}

function list_memcfl() {
    var list_ptr = _memcfl_list_all();
    var list_count = new Int32Array(Module.HEAPU8.buffer, list_ptr, 1)[0];
    var list = new Int32Array(Module.HEAPU8.buffer, list_ptr+4, list_count);
    var files = [];
    for(var i=0;i<list_count;i++) {
        var ptr = list[i];
        if (ptr==0) {continue;}
        var name = "";
        for(;Module.HEAPU8[ptr]!=0&&Module.HEAPU8[ptr]!=undefined;ptr++) {
            name += String.fromCharCode(Module.HEAPU8[ptr]);
        }
        files.push(name);   
    }
    display_memcfl_files(files);
    return files;
}

function unlink_memcfl(name) {
    heapName = allocFromString(name);
    heapName_byteOffset = heapName.byteOffset;
    _memcfl_unlink(heapName_byteOffset);
    free(heapName);
    list_memcfl();
}

function allocFromString(string) {
    var heapArray = alloc(string.length+1);
    heapArray.fill(0);
    for(var i=0;i<string.length;i++) {
        heapArray[i] = string.charCodeAt(i);
    }
    return heapArray;
}

function allocFromStringArray(inArgv) {
    var heapArgv = alloc(inArgv.length*4);
    var heapArgv32 = new Int32Array(Module.HEAPU8.buffer, heapArgv.byteOffset, inArgv.length);
    for(var k in inArgv) {
        var heapArray = allocFromString(inArgv[k]);
        var heapArray_byteOffset = heapArray.byteOffset;
        heapArgv32[k] = heapArray_byteOffset;
    }
    
    return heapArgv;
}

/** Create a heap array from the array ar. */
function allocFromArray(ar) {
    /* Allocate */
    var nbytes = ar.length * ar.BYTES_PER_ELEMENT;
    var heapArray = alloc(nbytes);

    /* Copy */
    heapArray.set(new Uint8Array(ar.buffer));
    return heapArray;
}

/** Allocate a heap array to be passed to a compiled function. */
function alloc(nbytes) {
    var ptr = Module._malloc(nbytes)>>>0;
    return new Uint8Array(Module.HEAPU8.buffer, ptr, nbytes);
}

/** Free a heap array. */
function free(heapArray) {
    Module._free(heapArray.byteOffset);
}
