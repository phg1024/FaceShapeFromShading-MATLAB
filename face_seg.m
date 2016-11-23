function face_seg(path)
close all;

all_images = read_settings(fullfile(path, 'settings.txt'));

core_face_mask = logical(im2double(imread('/home/phg/Data/Multilinear/albedos/core_face.png')));
hair_region_mask = logical(im2double(imread('/home/phg/Data/Multilinear/albedos/hair_mask.png')));
mean_texture = im2double(imread(fullfile(path, 'SFS', 'mean_texture.png')));

figure;imshow(mean_texture);
figure;
skin_pixels = build_color_model(core_face_mask, mean_texture, struct('newfig', false, 'marker', '.'));
hair_pixels = build_color_model(hair_region_mask, mean_texture, struct('newfig', false, 'marker', '+'));

X = [skin_pixels; hair_pixels];
Y = [ones(size(skin_pixels, 1), 1); ones(size(hair_pixels, 1), 1)*-1];
train_idx = randperm(size(X, 1), 5000);
X_train = X(train_idx,:);
Y_train = Y(train_idx,:);
%cl = fitcsvm(X_train, Y_train, ...
%             'KernelFunction', 'rbf', 'BoxConstraint', Inf, 'ClassNames', [-1, 1]);
cl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);

hair_model_gmm = fitgmdist(hair_pixels, 1);

d = 0.005;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)), min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[labels,scores] = predict(cl,xGrid);

% Plot the data and the decision boundary
figure;
gscatter(X(:,1),X(:,2),Y,'rb','.');
hold on

%h(3) = plot(X_train(cl.IsSupportVector,1),X_train(cl.IsSupportVector,2),'ko');
gscatter(x1Grid(:),x2Grid(:),labels, 'gy', 'x');
axis equal
hold off

%return;
mkdir(fullfile(path, 'masked'));

for i=1:length(all_images)
    close all;
    
    [~, basename, ext] = fileparts(all_images{i})
    
%     if strcmp(basename, '36') == 0
%         continue;
%     end
    
    input_image = fullfile(path, all_images{i});
    
    albedo_image = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', i-1));
    
    I = im2double(imread(input_image));
    Ib = I;
    %for j=1:1
    %    Ib = bfilter2(Ib, 7, [3 0.1]);
    %end
    
    [h, w, ~] = size(I);
    Ia = im2double(imread(albedo_image));
    figure;imshow(I);
    figure;imshow(Ia);
    figure;imshow(Ib);
    
    Imask = rgb2gray(Ia);
    Imask(Imask > 0) = 1.0;
    Imask_rgb(:,:,1) = Imask;
    Imask_rgb(:,:,2) = Imask;
    Imask_rgb(:,:,3) = Imask;
    figure;imshow(Imask);
    
    % use superpixels for clustering
    if true
        [L,N] = superpixels(I .* Imask_rgb, 1024);
        BW = boundarymask(L);
        figure;imshow(imoverlay(I .* Imask_rgb,BW .* Imask,'cyan'));
        
        A = I .* Imask_rgb;
        Iseg = zeros(size(A),'like',A);
        seg_idx = label2idx(L);
        seg_color = zeros(N, 3);
        numRows = size(A,1);
        numCols = size(A,2);
        for labelVal = 1:N
            redIdx = seg_idx{labelVal};
            greenIdx = seg_idx{labelVal}+numRows*numCols;
            blueIdx = seg_idx{labelVal}+2*numRows*numCols;
            avg_r = mean(A(redIdx));
            avg_g = mean(A(greenIdx));
            avg_b = mean(A(blueIdx));
            seg_color(labelVal,:) = [avg_r, avg_g, avg_b];
            Iseg(redIdx) = avg_r;
            Iseg(greenIdx) = avg_g;
            Iseg(blueIdx) = avg_b;
        end
        
        figure;imshow(Iseg);
    end
    
    Ycbcr = rgb2ycbcr(Ib);
    figure;imshow(Ycbcr);
    cb = Ycbcr(:,:,2);
    cr = Ycbcr(:,:,3);
    
    Ycbcr_a = rgb2ycbcr(Ia);
    cb_a = Ycbcr_a(:,:,2);
    cr_a = Ycbcr_a(:,:,3);
    
    Ycbcr_seg = rgb2ycbcr(seg_color);
    cb_seg = Ycbcr_seg(:,2);
    cr_seg = Ycbcr_seg(:,3);
    
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
    
    if false
        % classifiy the good points
        good_pts_idx = find(good_pts>0);
        [labels, scores] = predict(cl, [cb(good_pts), cr(good_pts)]);
        good_pts_refined = good_pts;
        good_pts_refined(good_pts_idx(labels<0)) = 0;
        
        p_hair = pdf(hair_model_gmm, [cb(good_pts), cr(good_pts)]);
        p_hair = min(mahal(hair_model_gmm, [cb(good_pts), cr(good_pts)]), [], 2);
        size(p_hair)
        good_pts_hair_idx = find(p_hair>15);
        good_pts_hair = good_pts;
        good_pts_hair(good_pts_idx(good_pts_hair_idx)) = 0;
        
        p_hair = min(mahal(hair_model_gmm, [cb(:), cr(:)]), [], 2);
        size(p_hair)
        figure;imagesc(reshape(p_hair, 250, 250) .* Imask);axis equal;
        
        figure;
        subplot(1, 3, 1); imshow(good_pts);
        subplot(1, 3, 2); imshow(good_pts_refined);
        subplot(1, 3, 3); imshow(good_pts_hair);
        pause;
        
        % augment the classifier with this image
        Xi = [cb(good_pts), cr(good_pts)];
        Xir = [cb(~good_pts), cr(~good_pts)];
        Yi = ones(size(Xi,1), 1);
        Yir = -1 * ones(size(Xir,1), 1);
        XX = [X(ref_samples,:);Xi;Xir];
        YY = [Y(ref_samples,:);Yi;Yir];
        ref_samples = 1:size(X,1);
        cl = fitcknn(XX, YY,'NumNeighbors', 5, 'Standardize',1);
        figure;
        gscatter(XX(:,1), XX(:,2), YY, 'rb', '.');
        
        d = 0.0025;
        [x1Grid,x2Grid] = meshgrid(min(XX(:,1)):d:max(XX(:,1)), min(XX(:,2)):d:max(XX(:,2)));
        xGrid = [x1Grid(:),x2Grid(:)];
        [labels,scores] = predict(cl,xGrid);
        
        % Plot the data and the decision boundary
        figure;
        gscatter(XX(:,1),XX(:,2),YY,'rb','.');
        hold on
        
        %h(3) = plot(X_train(cl.IsSupportVector,1),X_train(cl.IsSupportVector,2),'ko');
        gscatter(x1Grid(:),x2Grid(:),labels, 'gy', '.');
        axis equal
        hold off
    end
    
    for i=1:5
        figure; subplot(1, 2, 1); hold on;
        scatter(cb(good_pts), cr(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled');
        axis equal;
        if i == 1
            init_S.mu = clustCent(:,1)';
            init_S.Sigma = eye(2);
            obj = fitgmdist([cb(good_pts), cr(good_pts); cb_seg, cr_seg], 1, 'Start', init_S);
        else
            obj = fitgmdist([cb(good_pts), cr(good_pts)], floor((i+1)/2));
        end
        limits = get(gca,{'XLim','YLim'});
        xlim = limits{1}
        ylim = limits{2}
        xc = mean(xlim); yc = mean(ylim);
        Lx = xlim(2) - xlim(1); Ly = ylim(2) - ylim(1);
        L = max([Lx, Ly]);
        ezcontour(@(x1,x2)pdf(obj,[x1 x2]), [xlim ylim]);
        colorbar;
        axis([xc-0.5*L xc+0.5*L yc-0.5*L yc+0.5*L]);
        
        p = pdf(obj, [cb(:), cr(:)]);
        p = min(mahal(obj, [cb(:), cr(:)]), [], 2);
        size(p)
        %[f, x] = ecdf(p);
        %figure;plot(x, f);
        good_p = p<(5/(i^0.1));
        good_p = reshape(good_p, h, w);
        
        pa = pdf(obj, [cb_a(:), cr_a(:)]);
        pa = min(mahal(obj, [cb_a(:), cr_a(:)]), [], 2);
        good_pa = pa<(5/(i^0.1));
        good_pa = reshape(good_pa, h, w);
        
        if false
        [labels, scores] = predict(cl, [cb(:), cr(:)]);
        good_ph = labels > 0;
        good_ph = reshape(good_ph, h, w);
        
        figure;
        num_masks = 4;
        subplot(1, num_masks, 1); imshow(I .* repmat(good_pts, [1, 1, 3]) + ones(size(I))*0.5 .* repmat(1-good_pts, [1, 1, 3]));
        subplot(1, num_masks, 2); imshow(I .* repmat(good_p, [1, 1, 3]) + ones(size(I))*0.5 .* repmat(1-good_p, [1, 1, 3]));
        subplot(1, num_masks, 3); imshow(I .* repmat(good_pa, [1, 1, 3]) + ones(size(I))*0.5 .* repmat(1-good_pa, [1, 1, 3]));
        subplot(1, num_masks, 4); imshow(I .* repmat(good_ph, [1, 1, 3]) + ones(size(I))*0.5 .* repmat(1-good_ph, [1, 1, 3]));
        pause;
        valid_pts = good_ph .* good_pa .* good_p .* good_pts;
        else
        %figure;scatter3(r(good_pts), g(good_pts), b(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled');axis equal;
        
        valid_pts = good_pa .* good_p .* good_pts;
        end
        %figure; imshow(valid_pts);
        
        % remove outlier segments as well
        p_seg = pdf(obj, [cb_seg, cr_seg]);
        p_seg = min(mahal(obj, [cb_seg, cr_seg]), [], 2);
        for i=1:N
            if p_seg(i) >= (15/i^0.1)
                valid_pts(seg_idx{i}) = 0;
            end
        end
        
        
        % shrink a bit
        se = strel('disk',4);
        valid_pts = imdilate(valid_pts, se);
        valid_pts = imfill(valid_pts, 'holes');
        valid_pts = imerode(valid_pts, se);
        valid_pts(~good_pts) = 0;
        %figure; imshow(valid_pts);
        
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
        
        subplot(1, 2, 2); imshow(Ifinal);
    end
    
    % finally, shrink a bit
    se = strel('disk',1);
    valid_pts = imerode(valid_pts, se);
    
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
    
    r = Ia(:,:,1);
    g = Ia(:,:,2);
    b = Ia(:,:,3);
    
    r(~valid_pts) = 0.5;
    g(~valid_pts) = 0.5;
    b(~valid_pts) = 0.5;
    
    Iafinal = Ia;
    Iafinal(:,:,1) = r;
    Iafinal(:,:,2) = g;
    Iafinal(:,:,3) = b;
    
    figure;
    set(gcf,'units','points','position',[-1000, 100, 800, 600]);
    num_cols = 5;
    subplot(1, num_cols, 1);imshow(I);
    subplot(1, num_cols, 2);imshow(I.*Imask_rgb);
    subplot(1, num_cols, 3);imshow(Iseg .* Imask_rgb);
    subplot(1, num_cols, 4);imshow(Ifinal);
    subplot(1, num_cols, 5);imshow(Iafinal);
    
    %imwrite(valid_pts, fullfile(path, [basename, '_mask.png']));
    imwrite(Ifinal, fullfile(path, 'masked', [basename, '.png']));
    imwrite(valid_pts, fullfile(path, 'masked', ['mask', basename, '.png']));
    
    %pause;
end

end