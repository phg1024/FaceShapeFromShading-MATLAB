function model = build_color_model(mask, I, options)
YCbCr = rgb2ycbcr(I);
Y0 = YCbCr(:,:,1);
Cb0 = YCbCr(:,:,2);
Cr0 = YCbCr(:,:,3);

r0 = I(:,:,1);
g0 = I(:,:,2);
b0 = I(:,:,3);

r = r0(mask);
g = g0(mask);
b = b0(mask);

y = Y0(mask);
cb = Cb0(mask);
cr = Cr0(mask);

if options.newfig
    figure;
else
    hold on;
end

scatter(cb(:), cr(:), 10.0, [r(:), g(:), b(:)], options.marker);
axis equal;

model = [cb(:), cr(:)];
end
