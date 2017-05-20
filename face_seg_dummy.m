function face_seg_dummy(path)
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

    % fall back solution
    Ifinal = I;
    valid_pts = Imask;

    % save the mask including eyebrow region
    imwrite(valid_pts, fullfile(path, 'masked', ['a_mask', basename, '.png']));

    Ifinal = Ifinal .* cat(3, valid_pts, valid_pts, valid_pts);

    % save the mask excluding eyebrow region
    imwrite(Ifinal, fullfile(path, 'masked', [basename, '.png']));
    imwrite(valid_pts, fullfile(path, 'masked', ['mask', basename, '.png']));
end

end
