function data = readcfl(filenameBase)
% function data = readcfl(filenameBase)
%
% Read in recon data stored in filenameBase.cfl (complex float)
% based on dimensions stored in filenameBase.hdr.

    dims = readReconHeader(filenameBase);

    filename = strcat(filenameBase,'.cfl');
    fid = fopen(filename);

    data_r_i = fread(fid, prod([2 dims]), 'float32');
    data_r_i = reshape(data_r_i, [2 dims]);
    data = zeros(dims);
    data(:) = data_r_i(1:2:end) + 1i*data_r_i(2:2:end);

    fclose(fid);
end

function dims = readReconHeader(filenameBase)
    filename = strcat(filenameBase,'.hdr');
    fid = fopen(filename);
    
    line = getNextLine(fid);
    dims = str2num(line);
    
    fclose(fid);
end

function line = getNextLine(fid)
    line = fgetl(fid);
    while(line(1) == '#')
        line = fgetl(fid);
    end
end
