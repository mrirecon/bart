function [varargout] = bart(cmd, varargin);
% BART	Call BART command from Matlab.
%   [A B] = bart('command', X Y) call command with inputs X Y and outputs A B
%
% 2014-2015 Martin Uecker <uecker@med.uni-goettingen.de>

	bart_path = getenv('TOOLBOX_PATH');

	if isempty(bart_path)
		if exist('/usr/local/bin/bart', 'file')
			bart_path = '/usr/local/bin';
		elseif exist('/usr/bin/bart', 'file')
			bart_path = '/usr/bin';
		else
			error('Environment variable TOOLBOX_PATH is not set.');
		end
	end

	% clear the LD_LIBRARY_PATH environment variable (to work around
	% a bug in Matlab).

	if ismac==1
		setenv('DYLD_LIBRARY_PATH', '');
	else
		setenv('LD_LIBRARY_PATH', '');
	end

	name = tempname;

	in = cell(1, nargin - 1);

	for i=1:nargin - 1,
		in{i} = strcat(name, 'in', num2str(i));
		writecfl(in{i}, varargin{i});
	end

	in_str = sprintf(' %s', in{:});

	out = cell(1, nargout);

	for i=1:nargout,
		out{i} = strcat(name, 'out', num2str(i));
	end

	out_str = sprintf(' %s', out{:});

	if ispc
		% For cygwin use bash and modify paths
		ERR = system(['bash.exe --login -c ', ...
			strrep(bart_path, filesep, '/'), ...
	                '"', '/bart ', strrep(cmd, filesep, '/'), ' ', ...
			strrep(in_str, filesep, '/'), ...
                	' ', strrep(out_str, filesep, '/'), '"']);
	else
		ERR = system([bart_path, '/bart ', cmd, ' ', in_str, ' ', out_str]);
	end

	if ERR~=0
		error('command exited with an error');
	end

	for i=1:nargin - 1,
		delete(strcat(in{i}, '.cfl'));
		delete(strcat(in{i}, '.hdr'));
	end

	for i=1:nargout,
		varargout{i} = readcfl(out{i});
		delete(strcat(out{i}, '.cfl'));
		delete(strcat(out{i}, '.hdr'));
	end
end
