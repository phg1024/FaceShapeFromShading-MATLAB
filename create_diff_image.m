function I = create_diff_image(dval, mask, minVal, maxVal)
h = 1.0 - max(min((dval-minVal)/(maxVal-minVal)/0.67, 1.0), 0.0);
s = ones(size(dval));
v = ones(size(dval));
hsv(:,:,1) = h * 0.67; hsv(:,:,2) = s; hsv(:,:,3) = v;
I = hsv2rgb(hsv);

r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);

r(mask<1) = 0.5;
g(mask<1) = 0.5;
b(mask<1) = 0.5;

I = cat(3, r, g, b);
end