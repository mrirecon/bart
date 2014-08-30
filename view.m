#! /usr/bin/octave -qf

addpath(strcat(getenv("TOOLBOX_PATH"), "/matlab"));
arg_list = argv();


data = squeeze(readcfl(arg_list{1}));
imshow3(abs(data), []);
pause;


