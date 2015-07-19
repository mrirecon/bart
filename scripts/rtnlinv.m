% 2015, Martin Uecker <uecker@eecs.berkeley.edu>
%
% Example script to use BART for the initial preprocessing
% (gridding) which is required - but not included - in the
% original Matlab RT-NLINV example. The example is for a 
% single frame, but this should also work in a similar way 
% for the RT-NLINV2 code which reconstructs a time-series 
% of images from highly undersampled data using temporal
% regularization.
%
% Links to the Matlab code can be found here:
% http://www.eecs.berkeley.edu/~uecker/toolbox.html
%
% References:
%
% Uecker M et al., Nonlinear Inverse Reconstruction for Real-time MRI
% of the Human Heart Using Undersampled Radial FLASH,
% MRM 63:1456-1462 (2010)
%
% Uecker M et al., Real-time magnetic resonance imaging at 20 ms
% resolution, NMR in Biomedicine 23: 986-994 (2010)
%

% data set is included in the IRGNTV example
A = load('radial_cardiac_25_projections.mat');

% re-format trajectory for BART
t = zeros(3, 256, 25);
t(1,:,:) = real(A.k) * 384.;
t(2,:,:) = imag(A.k) * 384.;

% use adjoint nufft to interpolate data onto Cartesia grid
adj = bart('nufft -d384:384:1 -a ', t, reshape(A.rawdata, [1 256 25 12]));

% compute point-spread function
psf = bart('nufft -d384:384:1 -a ', t, ones(1, 256, 25));

% transform back to k-space
adjk = bart('fft -u 7', adj);
psfk = bart('fft -u 7', psf);

% use nlinv from RT-NLINV (nlinv2) matlab package
R = nlinv(squeeze(adjk), squeeze(psfk) * 1., 9, 'noncart');

