function [hair_pixels, hair_pixels2, confidence] = find_hair_pixels(I, valid_pixels_indices)

% using HSV space
I_hsv = rgb2hsv(I);
Ih = I_hsv(:,:,1); Is = I_hsv(:,:,2); Iv = I_hsv(:,:,3);

idx = kmeans([Ih(valid_pixels_indices), Is(valid_pixels_indices), Iv(valid_pixels_indices)], 2);

size_c1 = length(idx(idx==1));
size_c2 = length(idx(idx==2));

if size_c1 < size_c2
    hair_cluster = 1;
else
    hair_cluster = 2;
end

hair_pixels = valid_pixels_indices(idx==hair_cluster);

% using Lab space
I_lab = rgb2lab(I);
Ia = I_lab(:,:,2);
Ib = I_lab(:,:,3);

idx = kmeans([Ia(valid_pixels_indices), Ib(valid_pixels_indices)], 2);
size_c1 = length(idx(idx==1));
size_c2 = length(idx(idx==2));

if size_c1 < size_c2
    hair_cluster = 1;
else
    hair_cluster = 2;
end
hair_pixels2 = valid_pixels_indices(idx==hair_cluster);

length(valid_pixels_indices)
L1 = length(hair_pixels)
L2 = length(hair_pixels2)
common_pixels = intersect(hair_pixels, hair_pixels2);
confidence = length(common_pixels) / L1

% [h, w, ~] = size(I);
% mask = zeros(h, w);
% mask(valid_pixels_indices) = 1;
% figure;imagesc(Ih .* mask);axis equal;title('Ih');
% figure;imagesc(Is .* mask);axis equal;title('Is');
% figure;imagesc(Iv .* mask);axis equal;title('Iv');
% figure;imagesc(Ia .* mask);axis equal;title('Ia');
% figure;imagesc(Ib .* mask);axis equal;title('Ib');
% pause;
end