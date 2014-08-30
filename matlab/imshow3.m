function imshow3(img,range,shape)
% imshow3(img, [ range, [shape )
%
% function to display series of images as a montage.
% 
% img - a 3D array representing a series of images
% range - window level (similarly to imshow
% shape - a 2x1 vector representing the shape of the motage
%
% Example: 
% 		im = repmat(phantom(128),[1,1,6]);
%		figure;
%		imshow3(im,[],[2,3]);
%
% (c) Michael Lustig 2012

[sx,sy,nc] = size(img);

if nargin < 2 
	range = [min(img(:)), max(img(:))];
end

if isempty(range)==1
	range = [min(img(:)), max(img(:))];
end

if nargin < 3

	if  ceil(sqrt(nc))^2 ~= nc;
   	   nc = ceil(sqrt(nc))^2;
   	   img(end,end,nc)=0;
	end


	img = reshape(img,sx,sy*nc);
	img = permute(img,[2,3,1]);
	img = reshape(img,sy*sqrt(nc),sqrt(nc),sx);
	img = permute(img,[3,2,1]);
	img = reshape(img,sx*sqrt(nc),sy*sqrt(nc));

else
	img = reshape(img,sx,sy*nc);
	img = permute(img,[2,3,1]);
	img = reshape(img,sy*shape(2),shape(1),sx);
	img = permute(img,[3,2,1]);
	img = reshape(img,sx*shape(1),sy*shape(2));
end

%imagesc(img,range); colormap(gray(256));axis('equal');
imshow(img,range);
