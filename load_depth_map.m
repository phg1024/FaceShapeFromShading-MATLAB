function depth = load_depth_map(filename, target_sz)
fid = fopen(filename, 'rb');
sz = fread(fid, 2, 'int');
h = sz(1)
w = sz(2)
raw_depth = fread(fid, h*w*3, 'double');
raw_depth = reshape(raw_depth, 3, []);
x = reshape(raw_depth(1,:), w, h)';
y = reshape(raw_depth(2,:), w, h)';
z = reshape(raw_depth(3,:), w, h)';
size(raw_depth)
depth(:,:,1) = x;
depth(:,:,2) = y;
depth(:,:,3) = z;
end