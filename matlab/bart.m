function [varargout] = bart(cmd, varargin)
% BART Call BART command from Matlab.
%   [varargout] = BART(cmd, varargin) to run given bart command (cmd) using the
%   data arrays/matrices passed as varargin.
%
%   [A, B] = BART('command', X, Y) call command with inputs X Y and outputs A B
%
%   To output a list of available bart commands simply run "bart". To
%   output the help for a specific bart command type "bart command -h".
%
% Parameters:
%   cmd:        Command to run as string (including non data parameters)
%   varargin:   Data arrays/matrices used as input
%
% Example:
%   bart traj -h
%   [reco] = bart('nufft -i traj', data) call nufft with inputs data and outputs reco
%
% Authors:
% 2014-2016 Martin Uecker <uecker@med.uni-goettingen.de>
% 2018 (Edited for WSL) Soumick Chatterjee <soumick.chatterjee@ovgu.de>
% 2020 Martin Krämer <martin.kraemer@med.uni-jena.de>
% 2022 Jon Tamir <jtamir.utexas.edu>

    % Check input variables
	if nargin==0 || isempty(cmd)
		fprintf('Usage: bart <command> <arguments...>\n\n');
		cmd = '';
	end

    % Check bart toolbox path
	[bart_path, isWSL] = get_bart_path();
	if isempty(bart_path)
		error('BART path not detected.');
	end

	% Clear the LD_LIBRARY_PATH environment variable (to work around a bug in Matlab).
    % Store original library path to be restored later.
	if ismac==1
        libPath = getenv('DYLD_LIBRARY_PATH');
		setenv('DYLD_LIBRARY_PATH', '');
    else
        libPath = getenv('LD_LIBRARY_PATH');
		setenv('LD_LIBRARY_PATH', '');
	end

    % Strip string arguments that were passed as varargin
    strArgsInd = cellfun(@ischar,varargin);
    strArgs = varargin(strArgsInd);
    dataArgs = varargin(~strArgsInd);    
    if (~isempty(strArgs)) % append to cmd
        cmd = sprintf('%s %s', cmd, sprintf('%s ', strArgs{:}));
        cmd(end) = [];
    end
    
    % Root path for temporary file
	name = tempname;

    % Files used for input
	in = cell(1, length(dataArgs));
	for iFile = 1:length(dataArgs)
		in{iFile} = strcat(name, 'in', num2str(iFile));
		writecfl(in{iFile}, dataArgs{iFile});
	end
	in_str = sprintf(' %s', in{:});

    % Files used for output
	out = cell(1, nargout);
	for iFile = 1:nargout
		out{iFile} = strcat(name, 'out', num2str(iFile));
	end
	out_str = sprintf(' %s', out{:});
    
    % Run bart
	if ispc % running windows?
        if isWSL
			% For WSL and modify paths
			in_strWSL = wslPathCorrection(in_str);
			out_strWSL =  wslPathCorrection(out_str);
			final_strWSL = ['wsl ', bart_path, '/bart ', cmd, ' ', in_strWSL, ' ', out_strWSL];
			ERR = system(final_strWSL);
        else
			% For cygwin use bash and modify paths
			ERR = system(['bash.exe --login -c ', ...
			strrep(bart_path, filesep, '/'), ...
			        '"', '/bart ', strrep(cmd, filesep, '/'), ' ', ...
			strrep(in_str, filesep, '/'), ...
			        ' ', strrep(out_str, filesep, '/'), '"']);
        end
    else
		ERR = system([bart_path, '/bart ', cmd, ' ', in_str, ' ', out_str]);
	end

    % Remove input files
	for iFile = 1:length(in)
        if (exist(strcat(in{iFile}, '.cfl'),'file'))
			delete(strcat(in{iFile}, '.cfl'));
        end

        if (exist(strcat(in{iFile}, '.hdr'),'file'))
			delete(strcat(in{iFile}, '.hdr'));
        end
	end

    % Remove output files
	for iFile = 1:length(out)
        if ERR == 0
			varargout{iFile} = readcfl(out{iFile});
        end
        if (exist(strcat(out{iFile}, '.cfl'),'file'))
			delete(strcat(out{iFile}, '.cfl'));
        end
        if (exist(strcat(out{iFile}, '.hdr'),'file'))
			delete(strcat(out{iFile}, '.hdr'));
        end
	end   
    
    % Restore Library Path to it's original value
    if (~isempty(libPath))
        if ismac==1
            setenv('DYLD_LIBRARY_PATH', libPath);
        else
            setenv('LD_LIBRARY_PATH', libPath);
        end
    end
    
    % Check if running BART was successful
    if (ERR~=0) && (~isempty(cmd))
        error('command exited with an error');
    end
end

