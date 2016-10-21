function save_depth_map(filename, depth)
fid = fopen(filename, 'wb');
[h, w, ~] = size(depth);
fwrite(fid, [h, w], 'int');
x = depth(:,:,1)';
y = depth(:,:,2)';
z = depth(:,:,3)';

raw_depth = [x(:), y(:), z(:)]';
fwrite(fid, raw_depth, 'double');
end