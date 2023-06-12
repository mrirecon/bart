
const DIMS = 16;
const scalar_size = 4;

function show_phantom() {
    bart_command(["phantom", "-x 128", "phantom.mem"]);
    var [data, dims] = from_memcfl("phantom.mem");
    console.log("dims", dims);
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))

    var canvas = document.getElementById("phant");
    ctx = canvas.getContext('2d');
    canvas.width = dims[0];
    canvas.height = dims[1];
    idata = ctx.createImageData(dims[0], dims[1]);
    var result = new Uint8ClampedArray(dims[0] * dims[1] * 4);
    for(var x=0;x<dims[0];x++) {
        for(var y=0;y<dims[1];y++) {
            var i = y + x*dims[1];
            var j = x + y*dims[0];
            result[4*i] = data[2*j]*256;
            result[4*i+1] = data[2*j]*256;
            result[4*i+2] = data[2*j]*256;
            result[4*i+3] = 255;
        }
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);

    var btn = document.getElementById("fft_phantom");
    btn.removeAttribute("disabled");
}

function fft_phantom_direct() {
    var t = performance.now();
    var dims = new Int32Array(DIMS);
    dims.fill(1);
    dims[0] = 128;
    dims[1] = 128;
    phant_data = calc_phantom(dims);
    data = fft(phant_data, dims, 3);
    console.log("runtime direct: ", performance.now()-t);
    //var [phant_data, data] = calc_phantom_fft(dims);
    console.log(phant_data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
    var canvas = document.getElementById("fft_phant");
    ctx = canvas.getContext('2d');
    canvas.width = 128;
    canvas.height = 128;
    idata = ctx.createImageData(128, 128);
    var result = new Uint8ClampedArray(128 * 128 * 4);
    for(var i=0;i<128*128;i++) {
        result[4*i] = data[2*i]*256;
        result[4*i+1] = data[2*i]*256;
        result[4*i+2] = data[2*i]*256;
        result[4*i+3] = 255;
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);
}

function fft_phantom() {
    bart_command(["fft", "3", "phantom.mem", "kspace.mem"]);
    var [data, dims] = from_memcfl("kspace.mem");
    console.log("dims", dims);
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))

    var canvas = document.getElementById("fft_phant");
    ctx = canvas.getContext('2d');
    canvas.width = dims[0];
    canvas.height = dims[1];
    idata = ctx.createImageData(dims[0], dims[1]);
    var result = new Uint8ClampedArray(dims[0] * dims[1] * 4);
    for(var x=0;x<dims[0];x++) {
        for(var y=0;y<dims[1];y++) {
            var i = y + x*dims[1];
            var j = x + y*dims[0];
            result[4*i] = data[2*j]*256;
            result[4*i+1] = data[2*j]*256;
            result[4*i+2] = data[2*j]*256;
            result[4*i+3] = 255;
        }
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);

    var btn = document.getElementById("ifft_phantom");
    btn.removeAttribute("disabled");
}

function ifft_fft_phantom() {
    bart_command(["fft", "-i", "3", "kspace.mem", "out_img.mem"]);
    var [data, dims] = from_memcfl("out_img.mem");
    console.log("dims", dims);
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
    var size = 2;
    for(var dim in dims) {
        size = size*dims[dim];
    }
    for (i=0;i<size;i++) {
        data[i] /= 0.5*size;
    }
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
    
    var canvas = document.getElementById("ifft_phant");
    ctx = canvas.getContext('2d');
    canvas.width = dims[0];
    canvas.height = dims[1];
    idata = ctx.createImageData(dims[0], dims[1]);
    var result = new Uint8ClampedArray(dims[0] * dims[1] * 4);
    for(var x=0;x<dims[0];x++) {
        for(var y=0;y<dims[1];y++) {
            var i = y + x*dims[1];
            var j = x + y*dims[0];
            result[4*i] = data[2*j]*256;
            result[4*i+1] = data[2*j]*256;
            result[4*i+2] = data[2*j]*256;
            result[4*i+3] = 255;
        }
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);
}

function pics_phantom() {    
    //bart_command(["phantom", "-s 8", "-k", "-x 128", "pics_kspace.mem"]);
    bart_command(["phantom", "-s 8", "-x 128", "pics_phantom.mem"]);
    bart_command(["fft", "3", "pics_phantom.mem", "pics_kspace.mem"]);
    var [data, dims] = from_memcfl("pics_kspace.mem", dims);

    var canvas = document.getElementById("pics_kspace");
    ctx = canvas.getContext('2d');
    canvas.width = dims[0]*dims[3];
    canvas.height = dims[1];
    idata = ctx.createImageData(dims[0]*dims[3],dims[1]);
    var result = new Uint8ClampedArray(dims[0]*dims[3]*dims[1]*4);
    //for(var i=0;i<dims[0]*dims[1]*dims[4];i++) {
    for(var x=0;x<dims[0];x++) {
        for(var y=0;y<dims[1];y++) {
            for(var c=0;c<dims[3];c++) {
                var i = y + x*dims[1]*dims[3] + c*dims[1];
                var j = x + y*dims[0] + c*dims[0]*dims[1]
                result[4*i] = data[2*j]*256;
                result[4*i+1] = data[2*j]*256;
                result[4*i+2] = data[2*j]*256;
                result[4*i+3] = 255;
            }
        }
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);

    var btn = document.getElementById("btn_pics_ifft");
    btn.removeAttribute("disabled");
    var btn = document.getElementById("btn_pics");
    btn.removeAttribute("disabled");
}

function pics_ifft() {
    bart_command(["fft", "-i", "3", "pics_kspace.mem", "image_out.mem"]);
    var [data, dims] = from_memcfl("image_out.mem", dims);
    console.log("dims", dims);
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
    var size = 2;
    for(var dim in dims) {
        size = size*dims[dim];
    }
    for (i=0;i<size;i++) {
        data[i] /= 128*128*128*128*8;
    }
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))

    var canvas = document.getElementById("pics_ifft");
    ctx = canvas.getContext('2d');
    canvas.width = dims[0]*dims[3];
    canvas.height = dims[1];
    idata = ctx.createImageData(dims[0]*dims[3],dims[1]);
    var result = new Uint8ClampedArray(dims[0]*dims[3]*dims[1]*4);
    for(var x=0;x<dims[0];x++) {
        for(var y=0;y<dims[1];y++) {
            for(var c=0;c<dims[3];c++) {
                var i = y + x*dims[1]*dims[3] + c*dims[1];
                var j = x + y*dims[0] + c*dims[0]*dims[1];
                var d = data[2*j];
                //d /= (dims[0]*dims[1]);
                result[4*i] = d*256;
                result[4*i+1] = d*256;
                result[4*i+2] = d*256;
                result[4*i+3] = 255;
            }
        }
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);
}

function pics() {

    console.log("dims", dims);
    bart_command(["ecalib", "pics_kspace.mem", "sensitivities.mem"]);

    var [data, dims] = from_memcfl("sensitivities.mem", dims);
    console.log("dims", dims);
    
    var canvas = document.getElementById("pics_sens");
    ctx = canvas.getContext('2d');
    canvas.width = dims[0]*dims[4];
    canvas.height = dims[1];
    idata = ctx.createImageData(dims[0]*dims[4],dims[1]);
    var result = new Uint8ClampedArray(dims[0]*dims[4]*dims[1]*4);
    //for(var i=0;i<dims[0]*dims[1]*dims[4];i++) {
    for(var x=0;x<dims[0];x++) {
        for(var y=0;y<dims[1];y++) {
            for(var m=0;m<dims[4];m++) {
                var i = y + x*dims[1]*dims[4] + m*dims[1];
                var j = x + y*dims[0] + m*dims[0]*dims[1]
                result[4*i] = data[2*j]*256;
                result[4*i+1] = data[2*j]*256;
                result[4*i+2] = data[2*j]*256;
                result[4*i+3] = 255;
            }
        }
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);

    bart_command(["pics", "-l1", "-r 0.001", "pics_kspace.mem", "sensitivities.mem", "image_out.mem"]);
    
    var [data, dims] = from_memcfl("image_out.mem", dims);
    console.log("dims", dims);
    
    var canvas = document.getElementById("pics");
    ctx = canvas.getContext('2d');
    canvas.width = dims[0]*dims[4];
    canvas.height = dims[1];
    idata = ctx.createImageData(dims[0]*dims[4],dims[1]);
    var result = new Uint8ClampedArray(dims[0]*dims[4]*dims[1]*4);
    //for(var i=0;i<dims[0]*dims[1]*dims[4];i++) {
    for(var x=0;x<dims[0];x++) {
        for(var y=0;y<dims[1];y++) {
            for(var m=0;m<dims[4];m++) {
                var i = y + x*dims[1]*dims[4] + m*dims[1];
                var j = x + y*dims[0] + m*dims[0]*dims[1]
                result[4*i] = data[2*j]*256;
                result[4*i+1] = data[2*j]*256;
                result[4*i+2] = data[2*j]*256;
                result[4*i+3] = 255;
            }
        }
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);
}

function bart_command(argv) {
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
    console.log("Runtime", argv[0], performance.now()-t);
}

function calc_phantom(dims) {
    var sstrs = new Int32Array(DIMS);
    var samples = 0;
    var d3 = false;
    var kspace = false;
    var popts  = _pha_opts_defaults;

    for(var i=0;i<DIMS;i++) {sstrs[i] = 0;}
    
    var size = 2;
    for(var dim in dims) {
        size = size*dims[dim];
    }
    var data = alloc(size*scalar_size);
    var data_byteOffset = data.byteOffset;
    var heapDims = allocFromArray(dims);
    var heapDims_byteOffset = heapDims.byteOffset;
    var heapsstrs = allocFromArray(sstrs);
    var heapsstrs_byteOffset = heapsstrs.byteOffset;

    _num_init();
    var phant_sel = document.getElementById("select_phantom").value;
    switch(phant_sel){ 
        case "0":
            _calc_phantom(heapDims_byteOffset, data_byteOffset, d3, kspace, heapsstrs_byteOffset, samples, popts);
            break;
        case "1":
            _calc_bart(heapDims.byteOffset, data.byteOffset, kspace, heapsstrs.byteOffset, samples, popts);
            break;
        case "2":
            _calc_circ(heapDims.byteOffset, data.byteOffset, d3, kspace, heapsstrs.byteOffset, samples, popts);
            break;
    } 

    var pdata = new Float32Array(size);
    pdata.set(new Float32Array(Module.HEAPU8.buffer, data_byteOffset, size));

    free(data);
    free(heapDims);
    free(heapsstrs);
    return pdata;
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

/** Compute the FFT of a real-valued mxn matrix. */
function fft(data, dims, flags=0) {
    /* Allocate input and output arrays on the heap. */
    
    var size = 2
    for(var dim in dims) {
        size = size*dims[dim];
    }
    var outData = alloc(size*scalar_size);
    var outData_byteOffset = outData.byteOffset;
    var inData = allocFromArray(data);
    var inData_byteOffset = inData.byteOffset;
    
    var heapDims = allocFromArray(dims);
    var heapDims_byteOffset = heapDims.byteOffset

    _fftc(dims.length, heapDims_byteOffset, flags, outData_byteOffset, inData_byteOffset);

    /* Get spectrum from the heap, copy it to local array. */
    var spectrum = new Float32Array(size);
    if(scalar_size==8) {
        var tmp = new Float64Array(Module.HEAPU8.buffer, outData_byteOffset, size);
        for(var i=0;i<tmp.length;i++) { spectrum[i] = tmp[i]; }
    } else {
        spectrum.set(new Float32Array(Module.HEAPU8.buffer, outData_byteOffset, size));
    }

    /* Free heap objects. */
    free(inData_byteOffset);
    free(outData_byteOffset);
    free(heapDims_byteOffset);

    return spectrum;
}

/** Compute the inverse FFT of a real-valued mxn matrix. */
function ifft(data, dims, flags=0) {
    var size = 2
    for(var dim in dims) {
        size = size*dims[dim];
    }
    var outData = alloc(size*scalar_size);
    var outData_byteOffset = outData.byteOffset;
    var inData = allocFromArray(data);
    var inData_byteOffset = inData.byteOffset;

    var heapDims = allocFromArray(dims);
    var heapDims_byteOffset = heapDims.byteOffset;

    _ifftc(dims.length, heapDims_byteOffset, flags, outData_byteOffset, inData_byteOffset);

    var data = scalar_size==4 ? Float32Array.from(new Float32Array(Module.HEAPU8.buffer,outData_byteOffset, size)): Float32Array.from(new Float64Array(Module.HEAPU8.buffer,outData_byteOffset, size));

    for (i=0;i<size;i++) {
        data[i] /= 0.5*size;
    }

    free(inData);
    free(outData);
    free(heapDims);

    return data;
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
