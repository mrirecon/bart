
const DIMS = 16;
const scalar_size = 4;

function show_phantom() {
    var canvas = document.getElementById("phant");
    ctx = canvas.getContext('2d');
    canvas.width = 128;
    canvas.height = 128;
    idata = ctx.createImageData(128, 128);
    var dims = new Int32Array(DIMS);
    dims.fill(1);
    dims[0] = 128;
    dims[1] = 128;
    data = calc_phantom(dims);
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
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
    var canvas = document.getElementById("fft_phant");
    ctx = canvas.getContext('2d');
    canvas.width = 128;
    canvas.height = 128;
    idata = ctx.createImageData(128, 128);
    var dims = new Int32Array(DIMS);
    dims.fill(1);
    dims[0] = 128;
    dims[1] = 128;
    phant_data = calc_phantom(dims);
    data = fft(phant_data, dims, 3);
    //var [phant_data, data] = calc_phantom_fft(dims);
    console.log(phant_data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
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

function ifft_fft_phantom() {
    var canvas = document.getElementById("ifft_phant");
    ctx = canvas.getContext('2d');
    canvas.width = 128;
    canvas.height = 128;
    idata = ctx.createImageData(128, 128);
    var dims = new Int32Array(DIMS);
    dims.fill(1);
    dims[0] = 128;
    dims[1] = 128;
    phant_data = calc_phantom(dims);
    fft_data = fft(phant_data, dims, 3);
    data = ifft(fft_data, dims, 3);
    console.log(phant_data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
    console.log(fft_data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
    console.log(data.reduce((p,c,i) => i%2?[p[0], p[1]+c]:[p[0]+c,p[1]], [0,0]))
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
    //_calc_phantom(heapDims_byteOffset, data_byteOffset, d3, kspace, heapsstrs_byteOffset, samples, popts);
    _calc_bart(heapDims.byteOffset, data.byteOffset, kspace, heapsstrs.byteOffset, samples, popts);
    //_calc_circ(heapDims.byteOffset, data.byteOffset, d3, kspace, heapsstrs.byteOffset, samples, popts);

    var pdata = new Float32Array(size);
    pdata.set(new Float32Array(Module.HEAPU8.buffer, data_byteOffset, size));

    free(data);
    free(heapDims);
    free(heapsstrs);
    return pdata;
}

function calc_phantom_fft(dims) {
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
    var data_fft = alloc(size*scalar_size);
    var data_fft_byteOffset = data_fft.byteOffset;
    var heapDims = allocFromArray(dims);
    var heapDims_byteOffset = heapDims.byteOffset;
    var heapsstrs = allocFromArray(sstrs);
    var heapsstrs_byteOffset = heapsstrs.byteOffset;

    _num_init();
    //_calc_phantom(dims, data.byteOffset, d3, kspace, sstrs, samples, popts);
    _calc_bart_fft(heapDims_byteOffset, data_byteOffset, data_fft_byteOffset, kspace, heapsstrs_byteOffset, samples, popts);
    //_calc_circ(dims, data.byteOffset, d3, kspace, sstrs, samples, popts);

    var pdata = new Float32Array(size);
    pdata.set(new Float32Array(Module.HEAPU8.buffer, data_byteOffset, size));
    var fftdata = new Float32Array(size);
    fftdata.set(new Float32Array(Module.HEAPU8.buffer, data_fft_byteOffset, size));

    free(data);
    free(data_fft);
    free(heapDims);
    free(heapsstrs);
    return [pdata, fftdata];
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

    //for (i=0;i<size;i++) {
        //data[i] /= size;
    //}

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
