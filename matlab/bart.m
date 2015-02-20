function [varargout] = bart(cmd, varargin);
% BART	Call BART command from Matlab.
%   [A B] = bart('command', X Y) call command with inputs X Y and outputs A B
%
% 2014 Martin Uecker <uecker@eecs.berkeley.edu>

	if isempty(getenv('TOOLBOX_PATH'))
		error('Environment variable TOOLBOX_PATH is not set.');
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

	out = cell(1, nargout);

	for i=1:nargout,
		out{i} = strcat(name, 'out', num2str(i));
	end

	ERR = system([getenv('TOOLBOX_PATH'), '/', cmd, ' ', strjoin(in, ' '), ' ', strjoin(out, ' ')]);

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



