function hair_pixels = find_hair_pixels(I, valid_pixels_indices)

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
end