function writecfl(filenameBase,data)
% writecfl(filenameBase, data)
%    Writes recon data to filenameBase.cfl (complex float)
%    and write the dimensions to filenameBase.hdr.
%
%    Written to edit data for the Berkeley recon.
%
% 2012 Joseph Y Cheng (jycheng@mrsrl.stanford.edu).

    dims = size(data);
    writeReconHeader(filenameBase,dims);

    filename = strcat(filenameBase,'.cfl');
    fid = fopen(filename,'w');
    
    data_o = zeros(prod(dims)*2,1);
    data_o(1:2:end) = real(data(:));
    data_o(2:2:end) = imag(data(:));
    
    fwrite(fid,data_o,'float32');
    fclose(fid);
end

function writeReconHeader(filenameBase,dims)
    filename = strcat(filenameBase,'.hdr');
    fid = fopen(filename,'w');
    fprintf(fid,'# Dimensions\n');
    for N=1:length(dims)
        fprintf(fid,'%d ',dims(N));
    end
    if length(dims) < 5
        for N=1:(5-length(dims))
            fprintf(fid,'1 ');
        end
    end
    fprintf(fid,'\n');
    
    fclose(fid);
end

