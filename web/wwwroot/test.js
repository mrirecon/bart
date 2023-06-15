
const DIMS = 16;
const scalar_size = 4;

function show_phantom() {
    var canvas = document.getElementById("phant");
    ctx = canvas.getContext('2d');
    canvas.width = 128;
    canvas.height = 128;
    idata = ctx.createImageData(128, 128);
    var dims = Array(DIMS);
    dims[0] = 128;
    dims[1] = 128;
    data = calc_phantom(dims);
    console.log(data.reduce((p,c) => p+c),data.reduce((p,c) => p<c?p:c),data.reduce((p,c) => p<c?c:p))
    var result = new Uint8ClampedArray(128 * 128 * 4);
    for(var i=0;i<128*128;i++) {
        result[i] = data[2*i]*256;
        result[i+1] = data[2*i]*256;
        result[i+2] = data[2*i]*256;
        result[i+3] = 255;
    }
    idata.data.set(result);
    ctx.putImageData(idata, 0, 0);
}

function calc_phantom(dims) {
    var sstrs = Array(DIMS);
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

    _num_init();
    //_calc_phantom(dims, data.byteOffset, d3, kspace, sstrs, samples, popts);
    _calc_bart(dims, data.byteOffset, kspace, sstrs, samples, popts);
    //_calc_circ(dims, data.byteOffset, d3, kspace, sstrs, samples, popts);

    var pdata = new Float32Array(size);
    pdata.set(new Float32Array(Module.HEAPU8.buffer, data.byteOffset, size));

    free(data);
    return pdata;
}

/** Compute the FFT of a real-valued mxn matrix. */
function fft(data, dims, flags=0) {
    /* Allocate input and output arrays on the heap. */
    
    var size = 2
    for(var dim in dims) {
        size = size*dims[dim];
    }
    var outData = alloc(size*scalar_size);
    var inData = alloc(size*scalar_size);
    for(var i=0;i<data.length;i++) {
        inData[2*i] = data[i];
    }

    _fft(dims.length, dims, flags, inData.byteOffset, outData.byteOffset);

    /* Get spectrum from the heap, copy it to local array. */
    var spectrum = new Float32Array(size);
    if(scalar_size==8) {
        var tmp = new Float64Array(Module.HEAPU8.buffer, heapSpectrum.byteOffset, size);
        for(var i=0;i<tmp.length;i++) { spectrum[i] = tmp[i]; }
    } else {
        spectrum.set(new Float32Array(Module.HEAPU8.buffer, heapSpectrum.byteOffset, size));
    }   

    /* Free heap objects. */
    free(heapData);
    free(heapSpectrum);

    return spectrum;
}

/** Compute the inverse FFT of a real-valued mxn matrix. */
function ifft(data, dims, flags=0) {
    var size = 2
    for(var dim in dims) {
        size = size*dims[dim];
    }
    var outData = alloc(size*scalar_size);
    var inData = alloc(size*scalar_size);
    for(var i=0;i<data.length;i++) {
        inData[2*i] = data[i];
    }

    _ifft(dims.length, dims, flags, inData.byetOffset, outData.byteOffset);

    var data = scalar_size==4 ? Float32Array.from(new Float32Array(Module.HEAPU8.buffer,heapData.byteOffset, size)): Float32Array.from(new Float64Array(Module.HEAPU8.buffer,heapData.byteOffset, size));

    //for (i=0;i<size;i++) {
        //data[i] /= size;
    //}

    free(heapSpectrum);
    free(heapData);

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
