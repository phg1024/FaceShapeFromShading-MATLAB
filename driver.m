close all;

path = '/home/phg/Storage/Data/InternetRecon2/Zhang_Ziyi/crop';
path = '/home/phg/Storage/Data/InternetRecon0/yaoming/crop';

all_images = read_settings(fullfile(path, 'settings.txt'));

% LoG kernel and LoG matrix
[LoG, mat_LoG] = LoGMatrix(2, 256, 256, 1.0);

for i=1:length(all_images)
input_image = fullfile(path, all_images{i});

albedo_image = fullfile(path, 'SFS', sprintf('albedo%d.png', i-1));
normal_image = fullfile(path, 'SFS', sprintf('normal%d.png', i-1));
depth_map = fullfile(path, 'SFS', sprintf('depth_map%d.bin', i-1));

tic;
refined_normal_map = SFS(input_image, albedo_image, normal_image, depth_map, LoG, mat_LoG);
fprintf('image %d finished in %.3fs\n', i, toc);

pause(2);
end

return;

refined_normal_map = SFS('4.png', 'albedo4.png', 'normal4.png', LoG, mat_LoG);
