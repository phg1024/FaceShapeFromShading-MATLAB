function select_point_clouds(path)

visualize_results = false;
figure_idx = 1;

all_images = read_settings(fullfile(path, 'settings.txt'));
selected_images = read_indices(fullfile(path, 'multi_recon', 'selection.txt'));

all_diffs = load(fullfile(path, 'SFS', 'deformation.mat'));
all_diffs = all_diffs.all_diffs;

for i=1:length(all_images)
    %close all;

    [~, basename, ext] = fileparts(all_images{i})
    image_index = str2num(basename);
    all_basenames(i) = image_index;

    input_image_filename = fullfile(path, all_images{i});
    input_image{i} = imread(input_image_filename);
    albedo_image_filename = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', image_index));
    deformation_image_filename = fullfile(path, 'SFS', sprintf('deformation_%d.png', image_index));
    deformation_image{i} = imread(deformation_image_filename);

    % mask from albedo-based segmentation
    albedo_mask_filename = fullfile(path, 'masked', sprintf('a_mask%s.png', basename));
    albedo_mask = im2double(imread(albedo_mask_filename));

    % mask from deformation based region selection
    deformation_mask_filename = fullfile(path, 'SFS', sprintf('deformation_mask_%d.png', image_index));
    deformation_mask = im2double(imread(deformation_mask_filename));

    d = load_depth_map(fullfile(path, 'SFS', sprintf('depth_map%d.bin', image_index)));
    do = load_depth_map(fullfile(path, 'SFS', sprintf('optimized_depth_map_%d.bin', image_index)));

    % initial mask from large scale recon
    mask0 = d(:,:,3) < -10;
    mask0 = 1 - mask0;

    % mask from SFS
    mask = do(:,:,3) < -10;
    mask = 1 - mask;

    if visualize_results
        figure(figure_idx);

        img_idx = 1;
        subplot(1, 4, img_idx);imshow(mask0); title('mask0'); img_idx = img_idx + 1;
        subplot(1, 4, img_idx);imshow(mask); title('mask'); img_idx = img_idx + 1;
        subplot(1, 4, img_idx);imshow(albedo_mask); title('albedo mask'); img_idx = img_idx + 1;
        subplot(1, 4, img_idx);imshow(deformation_mask); title('deformation mask'); img_idx = img_idx + 1;

        figure_idx = figure_idx + 1;
    end

    % albedo / init
    init_valid_cnt = sum(sum(mask0));
    albedo_valid_cnt = sum(sum(albedo_mask));
    sfs_valid_cnt = sum(sum(mask));
    deformed_valid_cnt = sum(sum(deformation_mask));

    albedo_loss(i) = 1.0 - albedo_valid_cnt / init_valid_cnt;
    deformation_loss(i) = 1.0 - deformed_valid_cnt / sfs_valid_cnt;
    total_loss(i) = 1.0 - (albedo_valid_cnt / init_valid_cnt);% * (deformed_valid_cnt / sfs_valid_cnt);

    max_diff(i) = max(max(all_diffs{image_index}));
    mean_diff(i) = sum(sum(all_diffs{image_index})) / sfs_valid_cnt;

    if visualize_results
        pause;
    end
end

all_diffs
selected_images
albedo_loss
deformation_loss
total_loss
max_diff
mean_diff

final_metric = total_loss;% + max_diff * 2.0 + mean_diff * 5.0;

[~, sorted_idx] = sort(final_metric);

if visualize_results
    batch_size = 24;
    for k=1:ceil(length(all_images)/batch_size)
        figure;
        img_k_idx = 1;
        for i=(k-1)*batch_size+1:min(k*batch_size, length(all_images))
            j = sorted_idx(i);
            subplot(3, 8, img_k_idx); img_k_idx = img_k_idx + 1;
            imshow([input_image{j}; deformation_image{j}]);
            xlabel(sprintf('%f\n%f\n%f\n%f', total_loss(j), max_diff(j), mean_diff(j), final_metric(j)));
        end
    end
end

% pick the first 75% images
filtered_images = all_basenames(sorted_idx(1:0.8*length(sorted_idx)));
filtered_images = intersect(filtered_images, selected_images);

if visualize_results
    batch_size = 24;
    for k=1:ceil(length(filtered_images)/batch_size)
        figure;
        img_k_idx = 1;
        for i=(k-1)*batch_size+1:min(k*batch_size, length(filtered_images))
            j = filtered_images(i);
            subplot(3, 8, img_k_idx); img_k_idx = img_k_idx + 1;
            imshow([input_image{j}; deformation_image{j}]);
            xlabel(sprintf('%f\n%f\n%f\n%f', total_loss(j), max_diff(j), mean_diff(j), final_metric(j)));
        end
    end
end

filtered_images_indices = sort(filtered_images);
write_indices(fullfile(path, 'SFS', 'selection.txt'), filtered_images_indices);

end
