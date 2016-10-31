clear all; close all;
idx = 0;
I = im2double(imread(sprintf('%d.png', idx)));

[h, w, ~] = size(I);
Ia = im2double(imread(sprintf('albedo%d.png', idx)));
figure;imshow(I);
figure;imshow(Ia);

Imask = rgb2gray(Ia);
Imask(Imask > 0) = 1.0;
figure;imshow(Imask);

% use superpixels for clustering
[L,N] = superpixels(I .* Imask, 1000);
BW = boundarymask(L);
figure;imshow(imoverlay(I .* Imask,BW .* Imask,'cyan'));

Ycbcr = rgb2ycbcr(I);
figure;imshow(Ycbcr);
cb = Ycbcr(:,:,2);
cr = Ycbcr(:,:,3);
r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);

good_pts = logical(Imask);

if true
    bandwidth = 0.05;
    x = [cb(good_pts), cr(good_pts)]';
    [clustCent,point2cluster,clustMembsCell] = MeanShiftCluster(x, bandwidth);
    numClust = length(clustMembsCell);
    
    figure(10),clf,hold on
    scatter(cb(good_pts), cr(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled'); hold on;
    cVec = 'bgrcmykbgrcmykbgrcmykbgrcmyk';%, cVec = [cVec cVec];
    for k = 1:min(numClust,length(cVec))
        myMembers = clustMembsCell{k};
        myClustCen = clustCent(:,k);
        plot(x(1,myMembers),x(2,myMembers),[cVec(k) '.'])
        plot(myClustCen(1),myClustCen(2),'o','MarkerEdgeColor','k','MarkerFaceColor',cVec(k), 'MarkerSize',10)
    end
    title(['no shifting, numClust:' int2str(numClust)])
end

for i=1:3
    figure; subplot(1, 2, 1); hold on;
    scatter(cb(good_pts), cr(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled'); 
    axis equal;
    if i == 1
        init_S.mu = clustCent(:,1)';
        init_S.Sigma = eye(2);
        obj = fitgmdist([cb(good_pts), cr(good_pts)], 1, 'Start', init_S);
    else
        obj = fitgmdist([cb(good_pts), cr(good_pts)], floor((i+1)/2));
    end
    limits = get(gca,{'XLim','YLim'});
    xlim = limits{1}
    ylim = limits{2}
    xc = mean(xlim); yc = mean(ylim);
    Lx = xlim(2) - xlim(1); Ly = ylim(2) - ylim(1);
    L = max([Lx, Ly]);
    fcontour(@(x1,x2)pdf(obj,[x1 x2]), [xlim ylim]);
    colorbar;
    axis([xc-0.5*L xc+0.5*L yc-0.5*L yc+0.5*L]);
    
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
    se = strel('disk',4);
    valid_pts = imdilate(valid_pts, se);
    valid_pts = imfill(valid_pts, 'holes');
    valid_pts = imerode(valid_pts, se);
    %figure; imshow(valid_pts);
    valid_pts = valid_pts .* good_pts;
    
    good_pts = logical(valid_pts);
    
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
    subplot(1, 2, 2);imshow(Ifinal);    
end