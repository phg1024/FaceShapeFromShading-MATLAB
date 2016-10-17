function pixels = find_discontinuous_pixels(I_depth, thres)

z = I_depth(:,:,3);
z_edge = edge(z, 'sobel', thres);

se = strel('disk',1);
z_edge = imdilate(z_edge, se);

pixels = z_edge > 0;

figure; imshow(pixels); title('z edge');

end