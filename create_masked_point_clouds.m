clear; close all;

person = 'Andy_Lau';
path = sprintf('~/Storage/Data/InternetRecon2/%s/crop/', person);

%person = 'yaoming';
%path = sprintf('~/Storage/Data/InternetRecon0/%s/crop/', person);

all_images = read_settings(fullfile(path, 'settings.txt'));

for i=1:length(all_images)
    %close all;
    
    [~, basename, ext] = fileparts(all_images{i})
    
    input_image = fullfile(path, all_images{i});
    
    albedo_image = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', i-1));
    
    d = load_depth_map(fullfile(path, 'SFS', sprintf('depth_map%d.bin', i-1)));
    do = load_depth_map(fullfile(path, 'SFS', sprintf('optimized_depth_map_%d.bin', i-1)));
    figure(1);imagesc(d(:,:,3));axis equal;
    figure(2);imagesc(do(:,:,3));axis equal;
    mask = do(:,:,3) < -10;
    mask = 1 - mask;
    
    dorig = d(:,:,3) .* mask;
    dopt = do(:,:,3) .* mask;
    
    figure(3);
    subplot(1, 4, 1); imagesc(mask);axis equal; title('mask'); axis equal;
    subplot(1, 4, 2); imagesc(dorig);axis equal; title('orig'); axis equal;
    subplot(1, 4, 3); imagesc(dopt);axis equal; title('opt'); axis equal;
    subplot(1, 4, 4); imagesc(abs(dorig - dopt));axis equal; title('diff'); axis equal; caxis([0 0.05]); colorbar;
    
    diff = abs(dorig - dopt);
    diffimg = create_diff_image(diff, mask, 0, 0.15);
    
    figure(4); imshow(diffimg); title('diff'); axis equal;
    
    diffmask = diff < 0.05;
    diffmask = diffmask .* mask;
    figure(5); imshow(diffmask);
    
    masked_do = do;
    masked_dopt = do(:,:,3);    
    masked_dopt(diff >= 0.05) = -1e6;
    %masked_dopt(~mask) = -1e6;
    %figure;imagesc(masked_dopt);axis equal;title('masked depth');
    masked_do(:,:,3) = masked_dopt;
    
    imwrite(diffimg, fullfile(path, 'SFS', sprintf('deformation_%d.png', i-1)));
    imwrite(diffmask, fullfile(path, 'SFS', sprintf('deformation_mask_%d.png', i-1)));
    save_point_cloud(fullfile(path, 'SFS', sprintf('masked_optimized_point_cloud_%d.txt', i-1)), masked_do);
end