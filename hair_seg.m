function [hair_mask, hair_map] = hair_seg(I, Ia, pts)

Imask = rgb2gray(Ia);
Imask(Imask > 0) = 1.0;
Icut = I .* repmat(Imask, [1, 1, 3]);

Ihsv = rgb2hsv(I);
Is = Ihsv(:,:,2);
Iv = Ihsv(:,:,3);

[h, w, ~] = size(I);

hair_map = zeros(h, w);

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

% exclude core face region
core_face_region = convhull(pts(:,1), pts(:,2));
core_face_mask = poly2mask(pts(core_face_region,1), pts(core_face_region,2), h, w);

hair_mask = hair_mask .* (1 - core_face_mask);
end
