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
    
    I = im2double(imread(input_image));
    Ia = im2double(imread(albedo_image));
    pts = read_points(points_file);
    
    Imask = rgb2gray(Ia);
    Imask(Imask > 0) = 1.0;
    Icut = I .* repmat(Imask, [1, 1, 3]);
    
    Ihsv = rgb2hsv(I);
    Is = Ihsv(:,:,2);
    Iv = Ihsv(:,:,3);
    
    hair_map = zeros(h, w);
    [h, w, ~] = size(I);    
    wavelengths = [2, 3, 3.5];
    for j=1:length(wavelengths)
        wavelength_j = wavelengths(j);
        hair_maps{j} = zeros(h, w);
        for ori=0:5:180
            [mag_v, ~] = imgaborfilt(Iv, wavelength_j, ori);
            [mag_s, ~] = imgaborfilt(Iv, wavelength_j, ori);
            hair_maps{j} = max(hair_maps{j}, abs(mag_v) + abs(mag_s));
            %     figure(2);
            %     subplot(1,2,1);imshow(mag);
            %     subplot(1,2,2);imshow(phase);
            %     pause;
        end
        hair_map = hair_map + hair_maps{j} / max(hair_maps{j}(:));
    end

    hair_map = hair_map / length(wavelengths);
    hair_mask = hair_map > 0.1;
    
    figure(1);joint_plot(I, Ia, Imask, Icut, hair_map, hair_mask, hair_mask .* Icut);
    pause;
end