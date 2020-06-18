function data = readcfl(filenameBase)
%READCFL Read complex data from file.
%   READCFL(filenameBase) read in reconstruction data stored in filenameBase.cfl 
%   (complex float) based on dimensions stored in filenameBase.hdr.
%
%   Parameters:
%       filenameBase:   path and filename of cfl file (without extension)
%
%   Written to edit data with the Berkeley Advanced Reconstruction Toolbox (BART).
%
% Copyright 2016. CBClab, Maastricht University.
% 2016 Tim Loderhose (t.loderhose@student.maastrichtuniversity.nl)

    dims = readReconHeader(filenameBase);

    filename = strcat(filenameBase,'.cfl');
    fid = fopen(filename);

    data_r_i = fread(fid, prod([2 dims]), '*float32');
    data_r_i = reshape(data_r_i, [2 dims]);
    data = complex(zeros(dims,'single'),0);
    data(:) = complex(data_r_i(1,:),data_r_i(2,:));

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
