function save_point_cloud(filename, depth)
x = depth(:,:,1);
y = depth(:,:,2);
z = depth(:,:,3);

invalid_points = abs(z)>1e1;

x(invalid_points) = [];
y(invalid_points) = [];
z(invalid_points) = [];

fid = fopen(filename, 'w');
fprintf(fid, '%f %f %f\n', [x(:), y(:), z(:)]');
fclose(fid);
end