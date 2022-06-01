function [bart_path, isWSL] = get_bart_path()
% BART get BART for Matlab.
%   [bart_path, isWSL] = get_bart_path() will return the bart path as seen by Matlab,
%   and optionally will return whether WSL was detected.
%
% Authors:
% 2022 Jon Tamir <jtamir.utexas.edu>

	% Check bart toolbox path
	bart_path = getenv('TOOLBOX_PATH');
	isWSL = false;
	if isempty(bart_path)
		if exist('/usr/local/bin/bart', 'file')
			bart_path = '/usr/local/bin';
		elseif exist('/usr/bin/bart', 'file')
			bart_path = '/usr/bin';
		else
			% Try to execute bart inside wsl, if it works, then it returns status 0
			[bartstatus, ~] = system('wsl bart version -V');
			if bartstatus==0
				[~, bart_path] = system('wsl dirname $(which bart)');
				bart_path = strip(bart_path);
				isWSL = true;
			end
		end
	end
end

