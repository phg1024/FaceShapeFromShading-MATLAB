%% test hair segmentation
path = '~/Storage/Data/InternetRecon3/';
person = 'George_W_Bush';

path = [path, person];

[all_images, all_points] = read_settings(fullfile(path, 'settings.txt'));

for i=1:length(all_images)
    %close all;
    
    [~, basename, ext] = fileparts(all_images{i})
    
    input_image = fullfile(path, all_images{i});    
    albedo_image = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', i-1));    
    points_file = fullfile(path, all_points{i});

    if false
        figure(1);imshow(input_image);hold on;
        plot(pts(:,1), pts(:,2), 'g.');
        for j=1:size(pts, 1)
            text(pts(j,1), pts(j,2), num2str(j));
        end
        pause;
    end
    
    I = im2double(imread(input_image));
    Ia = im2double(imread(albedo_image));
    pts = read_points(points_file);
    
    Imask = rgb2gray(Ia);
    Imask(Imask > 0) = 1.0;
    Icut = I .* repmat(Imask, [1, 1, 3]);    
    
    [hair_mask, hair_map] = hair_seg(I, Ia, pts);
    
    figure(1);joint_plot(I, Ia, Imask, Icut, hair_map, hair_mask, hair_mask .* Icut);
    pause;
end