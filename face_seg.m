close all;
idx = 4;
I = im2double(imread(sprintf('%d.png', idx)));
[h, w, ~] = size(I);
Ia = im2double(imread(sprintf('albedo%d.png', idx)));
figure;imshow(I);
figure;imshow(Ia);

Ycbcr = rgb2ycbcr(I);
figure;imshow(Ycbcr);
cb = Ycbcr(:,:,2);
cr = Ycbcr(:,:,3);
r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);

Imask = rgb2gray(Ia);
Imask(Imask > 0) = 1.0;
figure;imshow(Imask);

good_pts = logical(Imask);

for i=1:5
    figure;
    scatter(cb(good_pts), cr(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled'); hold on;
    obj = fitgmdist([cb(good_pts), cr(good_pts)], floor((i+1)/2));
    fcontour(@(x1,x2)pdf(obj,[x1 x2]), cell2mat(get(gca,{'XLim','YLim'})));
    colorbar;
    
    p = pdf(obj, [cb(:), cr(:)]);
    p = min(mahal(obj, [cb(:), cr(:)]), [], 2);
    size(p)
    %[f, x] = ecdf(p);
    %figure;plot(x, f);
    good_p = p<(5/(i^0.1));
    good_p = reshape(good_p, h, w);
    
    %figure;scatter3(r(good_pts), g(good_pts), b(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled');axis equal;
    
    valid_pts = good_p .* good_pts;
    %figure; imshow(valid_pts);
    
    % shrink a bit
    se = strel('disk',3);
    valid_pts = imdilate(valid_pts, se);
    valid_pts = imfill(valid_pts, 'holes');
    valid_pts = imerode(valid_pts, se);
    %figure; imshow(valid_pts);
    
    good_pts = logical(valid_pts);
end

r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);

r(~valid_pts) = 0.5;
g(~valid_pts) = 0.5;
b(~valid_pts) = 0.5;

Ifinal = I;
Ifinal(:,:,1) = r;
Ifinal(:,:,2) = g;
Ifinal(:,:,3) = b;
figure;imshow(Ifinal);