#! /usr/bin/octave -qf

addpath(strcat(getenv("BART_TOOLBOX_PATH"), "/matlab"));
addpath(strcat(getenv("TOOLBOX_PATH"), "/matlab")); % support old environment variable
arg_list = argv();


data = squeeze(readcfl(arg_list{1}));
imshow3(abs(data), []);
pause;


