close all;

path = '/home/phg/Storage/Data/InternetRecon2/Zhang_Ziyi/crop';
%path = '/home/phg/Storage/Data/InternetRecon0/yaoming/crop';

% LoG kernel and LoG matrix
[LoG, mat_LoG] = LoGMatrix(2, 250, 250, 1.0);

all_images = read_settings(fullfile(path, 'settings.txt'));

for i=1:length(all_images)
input_image = fullfile(path, all_images{i});

albedo_image = fullfile(path, 'SFS', sprintf('albedo%d.png', i-1));
normal_image = fullfile(path, 'SFS', sprintf('normal%d.png', i-1));
depth_map = fullfile(path, 'SFS', sprintf('depth_map%d.bin', i-1));

%[h, w, ~] = size(imread(albedo_image));
%[LoG, mat_LoG] = LoGMatrix(2, h, w, 1.0);

tic;
refined_normal_map = SFS(input_image, albedo_image, normal_image, depth_map, LoG, mat_LoG);
fprintf('image %d finished in %.3fs\n', i, toc);

pause(2);
end

% TODO
% restore image scaling in ioutilites
% regenerate large scale
% prepare for SFS
% figure out how to generate mesh from SFS point cloud