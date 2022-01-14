%Soumick Chatterjee <soumick.chatterjee@ovgu.de>

function [outData] = wslPathCorrection(inData)
    outData=inData;
    for i = 'a':'z' %Replace drive letters with /mnt/
        outData=strrep(outData,[i,':'],['/mnt/',i]); %if drive letter is supplied in lowercase
        outData=strrep(outData,[upper(i),':'],['/mnt/',i]); %if drive letter is supplied as uppercase
    end
    outData = strrep(outData, '\', '/'); %Change windows filesep to linux filesep
end
