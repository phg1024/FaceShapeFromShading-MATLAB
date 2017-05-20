tic;
sfs_pipe = fopen('sfs_pipe', 'r');
path = fscanf(sfs_pipe, '%s');
fclose(sfs_pipe);

close all;
fprintf('The path of images is %s\n', path);

% create the parallel workers pool
poolobj = gcp('nocreate');
delete(poolobj);
parpool('8workers', 8);

% perform face segmentation first
face_seg(path);

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
options.path = path;

options.silent = true;

parfor i=1:length(all_images)
    input_image = fullfile(path, all_images{i})
    [~, basename, ~] = fileparts(all_images{i});
    albedo_image = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', i-1))
    normal_image = fullfile(path, 'SFS', sprintf('normal%d.png', i-1))
    mask_image = fullfile(path, 'masked', sprintf('mask%s.png', basename))
    depth_map = fullfile(path, 'SFS', sprintf('depth_map%d.bin', i-1))

    options_i = options;
    options_i.idx = i-1;

    %[h, w, ~] = size(imread(albedo_image));
    %[LoG, mat_LoG] = LoGMatrix(2, h, w, 1.0);

    tic;
    refined_normal_map = SFS(input_image, albedo_image, normal_image, depth_map, mask_image, options_i);
    fprintf('image %d finished in %.3fs\n', i, toc);
end

% create masks based on the refined point clouds
create_masked_point_clouds(path);

% select point clouds
select_point_clouds_dummy(path);

toc;
