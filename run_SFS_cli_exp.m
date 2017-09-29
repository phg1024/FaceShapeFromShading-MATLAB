sfs_pipe = fopen('sfs_pipe', 'r');
path = fgetl(sfs_pipe);
iteration_index = sscanf(fgetl(sfs_pipe), '%d');
fclose(sfs_pipe);

close all;
fprintf('The path of images is %s\n', path);
fprintf('Iteration %d\n', iteration_index);

all_images = read_settings(fullfile(path, 'settings.txt'));

in_img_for_size = imread(fullfile(path, all_images{1}));
[img_w, img_h, ~] = size(in_img_for_size)

% LoG kernel and LoG matrix
[LoG, mat_LoG] = LoGMatrix(2, img_w, img_h, 1.0);
[albedo_LoG, albedo_mat_LoG] = LoGMatrix(2, img_w, img_h, 0.5);

options.LoG = LoG;
options.mat_LoG = mat_LoG;
options.albedo_LoG = albedo_LoG;
options.albedo_mat_LoG = albedo_mat_LoG;
options.path = fullfile(path, ['iteration_', num2str(iteration_index)]);

options.silent = true;

parpool('8workers', 8);

parfor i=1:length(all_images)
    input_image = fullfile(path, all_images{i});
    [~, basename, ~] = fileparts(all_images{i});
    image_index = str2num(basename);
    albedo_image = fullfile(options.path, 'SFS', sprintf('albedo_transferred_%d.png', image_index));
    normal_image = fullfile(options.path, 'SFS', sprintf('normal%d.png', image_index));
    mask_image = fullfile(path, 'masked', sprintf('mask%d.png', image_index));
    depth_map = fullfile(options.path, 'SFS', sprintf('depth_map%d.bin', image_index));

    options_i = options;
    options_i.idx = image_index;

    %[h, w, ~] = size(imread(albedo_image));
    %[LoG, mat_LoG] = LoGMatrix(2, h, w, 1.0);

    tic;
    refined_normal_map = SFS(input_image, albedo_image, normal_image, depth_map, mask_image, options_i);
    fprintf('image %d finished in %.3fs\n', i, toc);
end

% create masks based on the refined point clouds
create_masked_point_clouds_exp(path, iteration_index);

% No need to do this in iterative steps
% select point clouds
% select_point_clouds_exp(path, iteration_index);
