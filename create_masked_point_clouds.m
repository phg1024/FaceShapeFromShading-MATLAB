function create_masked_point_clouds(path)
visualize_results = false;
all_images = read_settings(fullfile(path, 'settings.txt'));
all_diffs = cell(1024, 1);

for i=1:length(all_images)
    %close all;

    [~, basename, ext] = fileparts(all_images{i})
    image_index = str2num(basename);

    input_image = fullfile(path, all_images{i});

    albedo_image = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', image_index));

    % load Idiff
    load(fullfile(path, 'SFS', sprintf('error_%d.mat', image_index)));

    %fullfile(path, 'SFS', sprintf('depth_map%d.bin', i-1))
    d = load_depth_map(fullfile(path, 'SFS', sprintf('depth_map%d.bin', image_index)));
    %fullfile(path, 'SFS', sprintf('optimized_depth_map_%d.bin', i-1))
    do = load_depth_map(fullfile(path, 'SFS', sprintf('optimized_depth_map_%d.bin', image_index)));
    if visualize_results
        figure(1);imagesc(d(:,:,3));axis equal;
        figure(2);imagesc(do(:,:,3));axis equal;
    end
    mask = do(:,:,3) < -10;
    mask = 1 - mask;

    dorig = d(:,:,3) .* mask;
    dopt = do(:,:,3) .* mask;

    if visualize_results
        figure(3);
        subplot(1, 4, 1); imagesc(mask);axis equal; title('mask'); axis equal;
        subplot(1, 4, 2); imagesc(dorig);axis equal; title('orig'); axis equal;
        subplot(1, 4, 3); imagesc(dopt);axis equal; title('opt'); axis equal;
        subplot(1, 4, 4); imagesc(abs(dorig - dopt));axis equal; title('diff'); axis equal; caxis([0 0.05]); colorbar;
    end

    diff = abs(dorig - dopt);
    diffimg = create_diff_image(diff, mask, 0, 0.15);

    if visualize_results
        figure(4); imshow(diffimg); title('diff'); axis equal;
    end

    Idiffmask = imgaussfilt(Idiff, 1) < 0.020;
    if visualize_results
        figure(5);imshow(Idiffmask);
    end

    diffmask = diff < 0.05;
    diffmask = diffmask .* mask .* Idiffmask;
    if visualize_results
        figure(6); imshow(diffmask);
    end

    masked_do = do;
    masked_dopt = do(:,:,3);
    masked_dopt(diff >= 0.05) = -1e6;
    %masked_dopt(~mask) = -1e6;
    %figure;imagesc(masked_dopt);axis equal;title('masked depth');
    masked_do(:,:,3) = masked_dopt;

    % save deformation data
    all_diffs{image_index} = diff;

    imwrite(diffimg, fullfile(path, 'SFS', sprintf('deformation_%d.png', image_index)));
    imwrite(diffmask, fullfile(path, 'SFS', sprintf('deformation_mask_%d.png', image_index)));
    save_point_cloud(fullfile(path, 'SFS', sprintf('masked_optimized_point_cloud_%d.txt', image_index)), masked_do);
end

disp('saving deformation.mat');
save(fullfile(path, 'SFS', 'deformation.mat'), 'all_diffs');

end
