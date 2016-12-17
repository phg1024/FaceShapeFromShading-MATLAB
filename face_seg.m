function face_seg(path)
enable_visualization = false;
close all;

[all_images, all_points] = read_settings(fullfile(path, 'settings.txt'));

core_face_mask = logical(im2double(imread('/home/phg/Data/Multilinear/albedos/core_face.png')));
hair_region_mask = logical(im2double(imread('/home/phg/Data/Multilinear/albedos/hair_mask.png')));
mean_texture = im2double(imread(fullfile(path, 'SFS', 'mean_texture.png')));

if enable_visualization
    figure;imshow(mean_texture);
    figure;
end
skin_pixels = build_color_model(core_face_mask, mean_texture, ...
    struct('newfig', false, 'marker', '.', 'vis', enable_visualization));
hair_pixels = build_color_model(hair_region_mask, mean_texture, ...
    struct('newfig', false, 'marker', '+', 'vis', enable_visualization));

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

if enable_visualization
    % Plot the data and the decision boundary
    figure;
    gscatter(X(:,1),X(:,2),Y,'rb','.');
    hold on
    
    %h(3) = plot(X_train(cl.IsSupportVector,1),X_train(cl.IsSupportVector,2),'ko');
    gscatter(x1Grid(:),x2Grid(:),labels, 'gy', 'x');
    axis equal
    hold off
end

%return;
mkdir(fullfile(path, 'masked'));
Imask_rgb = cell(1, length(all_images));
init_S = cell(1, length(all_images));

parfor i=1:length(all_images)
    close all;
    
    [~, basename, ext] = fileparts(all_images{i})
    
    input_image = fullfile(path, all_images{i});
    
    albedo_image = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', i-1));
    
    input_points = fullfile(path, all_points{i});
    
    try
        points = read_points(input_points);
        
        I = im2double(imread(input_image));
        Ib = I;
        %for j=1:1
        %    Ib = bfilter2(Ib, 7, [3 0.1]);
        %end
        
        [h, w, ~] = size(I);
        Ia = im2double(imread(albedo_image));
        if enable_visualization
            figure;imshow(I);
            figure;imshow(Ia);
            figure;imshow(Ib);
        end
        
        Imask = rgb2gray(Ia);
        Imask(Imask > 0) = 1.0;
        Imask_rgb{i}(:,:,1) = Imask;
        Imask_rgb{i}(:,:,2) = Imask;
        Imask_rgb{i}(:,:,3) = Imask;
        if enable_visualization
            figure;imshow(Imask);
        end
        
        % use superpixels for clustering
        if true
            [L,N] = superpixels(I .* Imask_rgb{i}, 1024);
            BW = boundarymask(L);
            if enable_visualization
                figure;imshow(imoverlay(I .* Imask_rgb{i},BW .* Imask,'cyan'));
            end
            
            A = I .* Imask_rgb{i};
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
            
            if enable_visualization
                figure;imshow(Iseg);
            end
        end
        
        Ycbcr = rgb2ycbcr(Ib);
        if enable_visualization
            figure;imshow(Ycbcr);
        end
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
            
            if enable_visualization
                figure(10),clf,hold on
                scatter(cb(good_pts), cr(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled'); hold on;
            end
            cVec = 'bgrcmykbgrcmykbgrcmykbgrcmyk';%, cVec = [cVec cVec];
            for k = 1:min(numClust,length(cVec))
                myMembers = clustMembsCell{k};
                myClustCen = clustCent(:,k);
                if enable_visualization
                    plot(x(1,myMembers),x(2,myMembers),[cVec(k) '.'])
                    plot(myClustCen(1),myClustCen(2),'o','MarkerEdgeColor','k','MarkerFaceColor',cVec(k), 'MarkerSize',10)
                end
            end
            if enable_visualization
                title(['no shifting, numClust:' int2str(numClust)])
            end
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
            if enable_visualization
                figure;imagesc(reshape(p_hair, 250, 250) .* Imask);axis equal;
                
                figure;
                subplot(1, 3, 1); imshow(good_pts);
                subplot(1, 3, 2); imshow(good_pts_refined);
                subplot(1, 3, 3); imshow(good_pts_hair);
                pause;
            end
            
            % augment the classifier with this image
            Xi = [cb(good_pts), cr(good_pts)];
            Xir = [cb(~good_pts), cr(~good_pts)];
            Yi = ones(size(Xi,1), 1);
            Yir = -1 * ones(size(Xir,1), 1);
            XX = [X(ref_samples,:);Xi;Xir];
            YY = [Y(ref_samples,:);Yi;Yir];
            ref_samples = 1:size(X,1);
            cl = fitcknn(XX, YY,'NumNeighbors', 5, 'Standardize',1);
            if enable_visualization
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
        end
        
        for j=1:5
            if enable_visualization
                figure; subplot(1, 2, 1); hold on;
                scatter(cb(good_pts), cr(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled');
                axis equal;
            end
            
            if j == 1
                init_S{i}.mu = clustCent(:,1)';
                init_S{i}.Sigma = eye(2);
                obj = fitgmdist([cb(good_pts), cr(good_pts); cb_seg, cr_seg], 1, 'Start', init_S{i});
            else
                obj = fitgmdist([cb(good_pts), cr(good_pts)], floor((j+1)/2));
            end
            
            if enable_visualization
                limits = get(gca,{'XLim','YLim'});
                xlim = limits{1}
                ylim = limits{2}
                xc = mean(xlim); yc = mean(ylim);
                Lx = xlim(2) - xlim(1); Ly = ylim(2) - ylim(1);
                L = max([Lx, Ly]);
                ezcontour(@(x1,x2)pdf(obj,[x1 x2]), [xlim ylim]);
                colorbar;
                axis([xc-0.5*L xc+0.5*L yc-0.5*L yc+0.5*L]);
            end
            
            p = pdf(obj, [cb(:), cr(:)]);
            p = min(mahal(obj, [cb(:), cr(:)]), [], 2);
            size(p)
            %[f, x] = ecdf(p);
            %figure;plot(x, f);
            good_p = p<(5/(j^0.1));
            good_p = reshape(good_p, h, w);
            
            pa = pdf(obj, [cb_a(:), cr_a(:)]);
            pa = min(mahal(obj, [cb_a(:), cr_a(:)]), [], 2);
            good_pa = pa<(5/(j^0.1));
            good_pa = reshape(good_pa, h, w);
            
            if false
                [labels, scores] = predict(cl, [cb(:), cr(:)]);
                good_ph = labels > 0;
                good_ph = reshape(good_ph, h, w);
                
                if enable_visualization
                    figure;
                    num_masks = 4;
                    subplot(1, num_masks, 1); imshow(I .* repmat(good_pts, [1, 1, 3]) + ones(size(I))*0.5 .* repmat(1-good_pts, [1, 1, 3]));
                    subplot(1, num_masks, 2); imshow(I .* repmat(good_p, [1, 1, 3]) + ones(size(I))*0.5 .* repmat(1-good_p, [1, 1, 3]));
                    subplot(1, num_masks, 3); imshow(I .* repmat(good_pa, [1, 1, 3]) + ones(size(I))*0.5 .* repmat(1-good_pa, [1, 1, 3]));
                    subplot(1, num_masks, 4); imshow(I .* repmat(good_ph, [1, 1, 3]) + ones(size(I))*0.5 .* repmat(1-good_ph, [1, 1, 3]));
                    pause;
                end
                valid_pts = good_ph .* good_pa .* good_p .* good_pts;
            else
                %figure;scatter3(r(good_pts), g(good_pts), b(good_pts), 10.0, [r(good_pts), g(good_pts), b(good_pts)], 'filled');axis equal;
                
                valid_pts = good_pa .* good_p .* good_pts;
            end
            %figure; imshow(valid_pts);
            
            % remove outlier segments as well
            p_seg = pdf(obj, [cb_seg, cr_seg]);
            p_seg = min(mahal(obj, [cb_seg, cr_seg]), [], 2);
            for k=1:N
                if p_seg(k) >= (15/k^0.1)
                    valid_pts(seg_idx{k}) = 0;
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
            
            if enable_visualization
                subplot(1, 2, 2); imshow(Ifinal);
            end
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
        
        if enable_visualization
            figure;
            set(gcf,'units','points','position',[-1000, 100, 800, 600]);
            num_cols = 5;
            subplot(1, num_cols, 1);imshow(I);
            subplot(1, num_cols, 2);imshow(I.*Imask_rgb{i});
            subplot(1, num_cols, 3);imshow(Iseg .* Imask_rgb{i});
            subplot(1, num_cols, 4);imshow(Ifinal);
            subplot(1, num_cols, 5);imshow(Iafinal);
        end
    catch ME
        % fall back solution
        Ifinal = I;
        valid_pts = Imask;
    end
    
    % finally, exclude eye-brow region
    left_eye_brow = 22:27;
    right_eye_brow = 16:21;
        
    if enable_visualization
        figure; 
        subplot(1, 3, 1);
        imshow(I); hold on;
        plot(points(:,1), points(:,2), 'g.');
        for k=1:size(points,1)
            text(points(k,1), points(k,2), num2str(k));
        end

        fill(points(left_eye_brow,1), points(left_eye_brow,2), 'g');
        fill(points(right_eye_brow,1), points(right_eye_brow,2), 'g');
    end

    % find the eye-brow region
    [L,N] = superpixels(I .* Imask_rgb{i}, 8192);
    seg_idx = label2idx(L);
    all_eye_brow_pixels = [];
    for labelVal = 1:N
        % the pixels in this segment
        pixel_indices = seg_idx{labelVal};
        % include it if this segment overlaps with the eye-brow region 
        in_left = inpolygon(pixel_indices ./ h, mod(pixel_indices, h), points(left_eye_brow,1), points(left_eye_brow,2));
        in_right = inpolygon(pixel_indices ./ h, mod(pixel_indices, h), points(right_eye_brow,1), points(right_eye_brow,2));
        inside_thres = 0.1
        if sum(in_left) > length(in_left) * inside_thres || sum(in_right) > length(in_right) * inside_thres
            all_eye_brow_pixels = union(all_eye_brow_pixels, pixel_indices);
        end
    end
    length(all_eye_brow_pixels)

    Ieyebrow = zeros(h, w);
    Ieyebrow(all_eye_brow_pixels) = 255.0;
    %figure; imshow(Ieyebrow);

    se = strel('disk', 4, 8);
    Ieyebrow = imdilate(Ieyebrow, se);
    se = strel('disk', 2, 8);
    Ieyebrow = imerode(Ieyebrow, se);
    se = strel('disk', 4);
    Ieyebrow = imclose(Ieyebrow, se);
    Ieyebrow = imclose(Ieyebrow, se);
    Ieyebrow = imclose(Ieyebrow, se);
    %figure; imshow(Ieyebrow);
    %pause
    all_eye_brow_pixels = find(Ieyebrow > 0);
    
    Ieyebrow = ones(h, w);
    Iemask = Ieyebrow; Iemask(all_eye_brow_pixels) = 0;
    
    Ieyebrow = zeros(h, w);
    Ieyebrow(all_eye_brow_pixels) = 0.5; Ier = Ieyebrow;
    Ieyebrow(all_eye_brow_pixels) = 0.5; Ieg = Ieyebrow;
    Ieyebrow(all_eye_brow_pixels) = 1; Ieb = Ieyebrow;
        
    if enable_visualization
        subplot(1, 3, 2);
        imshow(I .* cat(3, Iemask) + cat(3, Ier, Ieg, Ieb));

        subplot(1, 3, 3);
        imshow(imoverlay(I .* Imask_rgb{i},BW .* Imask,'cyan'));
        pause;
    end
    
    valid_pts = valid_pts .* Iemask;
    Ifinal = Ifinal .* cat(3, valid_pts, valid_pts, valid_pts);
    
    %imwrite(valid_pts, fullfile(path, [basename, '_mask.png']));
    imwrite(Ifinal, fullfile(path, 'masked', [basename, '.png']));
    imwrite(valid_pts, fullfile(path, 'masked', ['mask', basename, '.png']));
    
    %pause;
end

end
