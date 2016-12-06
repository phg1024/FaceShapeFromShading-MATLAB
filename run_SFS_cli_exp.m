sfs_pipe = fopen('sfs_pipe', 'r');
path = fgetl(sfs_pipe);
iteration_index = sscanf(fgetl(sfs_pipe), '%d');
fclose(sfs_pipe);

close all;
fprintf('The path of images is %s\n', path);
fprintf('Iteration %d\n', iteration_index);

% LoG kernel and LoG matrix
[LoG, mat_LoG] = LoGMatrix(2, 250, 250, 1.0);
[albedo_LoG, albedo_mat_LoG] = LoGMatrix(2, 250, 250, 0.5);

options.LoG = LoG;
options.mat_LoG = mat_LoG;
options.albedo_LoG = albedo_LoG;
options.albedo_mat_LoG = albedo_mat_LoG;
options.path = fullfile(path, ['iteration_', num2str(iteration_index)]);

options.silent = true;

all_images = read_settings(fullfile(path, 'settings.txt'));

parpool(8);

parfor i=1:length(all_images)
input_image = fullfile(path, all_images{i});
[~, basename, ~] = fileparts(all_images{i});
albedo_image = fullfile(options.path, 'SFS', sprintf('albedo_transferred_%d.png', i-1));
normal_image = fullfile(options.path, 'SFS', sprintf('normal%d.png', i-1));
mask_image = fullfile(path, 'masked', sprintf('mask%s.png', basename));
depth_map = fullfile(options.path, 'SFS', sprintf('depth_map%d.bin', i-1));

options_i = options;
options_i.idx = i-1;

%[h, w, ~] = size(imread(albedo_image));
%[LoG, mat_LoG] = LoGMatrix(2, h, w, 1.0);

tic;
refined_normal_map = SFS(input_image, albedo_image, normal_image, depth_map, mask_image, options_i);
fprintf('image %d finished in %.3fs\n', i, toc);
end

% create masks based on the refined point clouds
create_masked_point_clouds_exp(path, iteration_index);
