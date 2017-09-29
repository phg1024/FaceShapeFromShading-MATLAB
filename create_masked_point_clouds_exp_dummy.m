function create_masked_point_clouds_exp(path, iteration_index)
visualize_results = false;

all_images = read_settings(fullfile(path, 'settings.txt'));
all_diffs = cell(1024, 1);

for i=1:length(all_images)
    %close all;

    [~, basename, ext] = fileparts(all_images{i})
    image_index = str2num(basename);

    input_image = fullfile(path, all_images{i});

    albedo_image = fullfile(path, ['iteration_', num2str(iteration_index)], 'SFS', sprintf('albedo_transferred_%d.png', image_index));

    d = load_depth_map(fullfile(path, ['iteration_', num2str(iteration_index)], 'SFS', sprintf('depth_map%d.bin', image_index)));
    do = load_depth_map(fullfile(path, ['iteration_', num2str(iteration_index)], 'SFS', sprintf('optimized_depth_map_%d.bin', image_index)));
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

    % Use dummy mask
    % diffmask = diff < 0.05;
    % diffmask = diffmask .* mask;
    % if visualize_results
    %   figure(5); imshow(diffmask);
    % end

    % This is a dummy mask
    diffmask = mask;

    masked_do = do;
    masked_dopt = do(:,:,3);
    % This is a dummy task, don't mask it
    %masked_dopt(diff >= 0.05) = -1e6;
    %masked_dopt(~mask) = -1e6;
    %figure;imagesc(masked_dopt);axis equal;title('masked depth');
    masked_do(:,:,3) = masked_dopt;

    all_diffs{image_index} = diff;

    imwrite(diffimg, fullfile(path, ['iteration_', num2str(iteration_index)], 'SFS', sprintf('deformation_%d.png', image_index)));
    imwrite(diffmask, fullfile(path, ['iteration_', num2str(iteration_index)], 'SFS', sprintf('deformation_mask_%d.png', image_index)));
    save_point_cloud(fullfile(path, ['iteration_', num2str(iteration_index)], 'SFS', sprintf('masked_optimized_point_cloud_%d.txt', image_index)), masked_do);
end

save(fullfile(path, ['iteration_', num2str(iteration_index)], 'SFS', 'deformation.mat'), 'all_diffs');

end
