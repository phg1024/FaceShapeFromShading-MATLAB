function depth = load_depth_map(filename, target_sz)
fid = fopen(filename, 'rb');
sz = fread(fid, 2, 'int');
h = sz(1);
w = sz(2);
raw_depth = fread(fid, h*w*3, 'double');
raw_depth = reshape(raw_depth, 3, []);
x = reshape(raw_depth(1,:), w, h)';
y = reshape(raw_depth(2,:), w, h)';
z = reshape(raw_depth(3,:), w, h)';
size(raw_depth)
if nargin < 2
    target_sz = [h, w]
end
depth(:,:,1) = imresize(x, target_sz, 'bilinear');
depth(:,:,2) = imresize(y, target_sz, 'bilinear');
depth(:,:,3) = imresize(z, target_sz, 'bilinear');

max(max(max(depth)))
min(min(min(depth)))
end