function [normal_map] = SFS_direct(input_file, albedo_file, normal_file, depth_file, mask_file, options)

if options.silent
    close all;
end

LoG = options.LoG;
mat_LoG = options.mat_LoG;

albedo_LoG = options.albedo_LoG;
albedo_mat_LoG = options.albedo_mat_LoG;

if isfield(options, 'downsample_factor')
    downsample_factor = options.dowsample_factor;
    need_downsample = true;
else
    need_downsample = false;
end

if isfield(options, 'normal_map')
    normal_map = options.normal_map;
    has_init_normal_map = true;
else
    has_init_normal_map = false;
end

I_input = im2double(imread(input_file));

I_albedo = im2double(imread(albedo_file));
I_normal = im2double(imread(normal_file));
I_mask = im2double(imread(mask_file));
I_normal_raw = imread(normal_file);

resize_to_input = true;

if resize_to_input
    [h, w, ~] = size(I_input);
else
    [h, w, ~] = size(I_albedo);
end

I_depth = load_depth_map(depth_file, [h, w]);

if need_downsample
    I_input = imresize(I_input, downsample_factor);
    [h, w, ~] = size(I_input);
    I_albedo = imresize(I_albedo, [h, w]);
    I_normal = imresize(I_normal, [h, w]);
    I_normal_raw = imresize(I_normal_raw, [h, w]);
    I_mask = imresize(I_mask, [h, w]);
end

if resize_to_input
    I_albedo = imresize(I_albedo, [h, w]);
    I_normal = imresize(I_normal, [h, w]);
    I_normal_raw = imresize(I_normal_raw, [h, w]);
    I_mask = imresize(I_mask, [h, w]);
else
    I_input = imresize(I_input, [h, w]);
end

% mask everything
[h, w, channels] = size(I_mask);
if channels == 1
    mask_1 = I_mask;
    I_mask(:,:,1) = mask_1; I_mask(:,:,2) = mask_1; I_mask(:,:,3) = mask_1;
end
I_albedo = I_albedo .* I_mask;
I_normal = I_normal .* I_mask;
I_normal_raw = uint8(double(I_normal_raw) .* I_mask);
mask_pixels = I_mask(:,:,1);
depth_val = I_depth(:,:,3);
depth_val(mask_pixels == 0) = -1e6;
I_depth(:,:,3) = depth_val;

figure;imshow(I_input);title('input');
figure;imshow(I_albedo);title('albedo');
figure;imshow(I_normal);title('normal');
plot_depth(I_depth, true);

% get the valid pixels
Inx = I_normal_raw(:,:,1); Iny = I_normal_raw(:,:,2); Inz = I_normal_raw(:,:,3);
valid_pixel_indices = intersect(intersect(find(Inx~=0), find(Iny~=0)), find(Inz~=0));
edge_pixel_indices = find_edge(I_normal_raw);
discontinuous_pixels_indices = find_discontinuous_pixels(I_depth, 0.025);
[hair_pixel_indices, hair_pixel_indices2] = find_hair_pixels(I_input, valid_pixel_indices);
valid_pixel_indices = setdiff(valid_pixel_indices, edge_pixel_indices);
%valid_pixel_indices = setdiff(valid_pixel_indices, find(discontinuous_pixels_indices>0));

lighting_pixel_indices = setdiff(valid_pixel_indices, hair_pixel_indices);
size(valid_pixel_indices)
num_pixels = length(valid_pixel_indices);

valid_pixel_mask = zeros(h, w);
valid_pixel_mask(valid_pixel_indices) = 1;
figure;imshow(valid_pixel_mask);title('valid mask');

plot_mask('hair pixels - Hsv', hair_pixel_indices, h, w);
plot_mask('hair pixels - Lab', hair_pixel_indices2, h, w);

% initialize normal
if ~has_init_normal_map
    init_normal_map = (I_normal - 0.5) * 2.0;
    normal_map = init_normal_map;
else
    normal_map = (normal_map - 0.5) * 2.0;
    init_normal_map = imresize(normal_map, [h, w]);
end

nx0 = init_normal_map(:, :, 1);
ny0 = init_normal_map(:, :, 2);
nz0 = init_normal_map(:, :, 3);

%normal_mag_map = nx0.*nx0 + ny0.*ny0 + nz0.*nz0;
%normal_mag_map(normal_mag_map > 1.1) = -1;
%figure;imagesc(normal_mag_map);title('normal mag');axis equal;colorbar;pause;

% normals
nx = nx0(:); nx = max(-1.0, min(1.0, nx));
ny = ny0(:); ny = max(-1.0, min(1.0, ny));
nz = nz0(:); nz = max(-1.0, min(1.0, nz));

% albedo and input pixels
ar = I_albedo(:,:,1); Ir = I_input(:,:,1);
ag = I_albedo(:,:,2); Ig = I_input(:,:,2);
ab = I_albedo(:,:,3); Ib = I_input(:,:,3);

albedo_map_LoG = imfilter(I_albedo, albedo_LoG, 'replicate');
% figure;imshow(albedo_map_LoG); title('albedo LoG');

LoG_ar = albedo_map_LoG(:,:,1);
LoG_ag = albedo_map_LoG(:,:,2);
LoG_ab = albedo_map_LoG(:,:,3);

normal_map_LoG = imfilter(init_normal_map, LoG, 'replicate');
% figure;imshow(normal_map_LoG); title('normal LoG');

LoG_nx_ref = normal_map_LoG(:,:,1);
LoG_ny_ref = normal_map_LoG(:,:,2);
LoG_nz_ref = normal_map_LoG(:,:,3);

%% some preparation
good_indices = zeros(h*w*3, 1);
good_indices(valid_pixel_indices) = 1;
good_indices(valid_pixel_indices+h*w) = 1;
good_indices(valid_pixel_indices+h*w*2) = 1;
bad_indices = 1 - good_indices;
good_indices = logical(good_indices);
bad_indices = logical(bad_indices);

good_indices_1 = zeros(h*w,1);
good_indices_1(valid_pixel_indices) = 1;
bad_indices_1 = 1 - good_indices_1;
good_indices_1 = logical(good_indices_1);
bad_indices_1 = logical(bad_indices_1);
size(valid_pixel_indices)
good_indices_2 = zeros(h*w*2,1);
good_indices_2(valid_pixel_indices) = 1;
good_indices_2(valid_pixel_indices+h*w) = 1;
good_indices_2 = logical(good_indices_2);

x0 = I_depth(:,:,1);
y0 = I_depth(:,:,2);
z0 = I_depth(:,:,3);
valid_depth_points = z0(:)>-1e5;
%z0(~valid_depth_points) = -1;
z = z0;

%figure;imshow(reshape(valid_depth_points, h, w) - valid_pixel_mask);pause;

pixel_indices = reshape(1:h*w, h, w);
pixel_indices_shift_up = circshift(pixel_indices, -1);    % -1
pixel_indices_shift_right = circshift(pixel_indices, [0, 1]);   % -1
v_pixel_indices = pixel_indices(:);
v_pixel_indices_shift_up = pixel_indices_shift_up(:);
v_pixel_indices_shift_right = pixel_indices_shift_right(:);

C1 = sparse(1:h*w, v_pixel_indices, ones(h*w, 1)) - sparse(1:h*w, v_pixel_indices_shift_up, ones(h*w, 1));
C2 = sparse(1:h*w, v_pixel_indices, ones(h*w, 1)) - sparse(1:h*w, v_pixel_indices_shift_right, ones(h*w, 1));

C_x = -C2; C_y = -C1;

% % replace normal with the one estimated from depth
% subplot(1,2,1);draw_normal(normal_map);title('input normal');
%
nx_d = C_x * z(:) ./ abs(C_x * x0(:));
ny_d = C_y * z(:) ./ abs(C_y * y0(:));
nz_d = ones(h*w,1);

normal_norm = sqrt(nx_d.^2 + ny_d.^2 + 1);
% nx = nx ./ normal_norm;
% ny = ny ./ normal_norm;
% nz = nz ./ normal_norm;
%
% normal_map = zeros(h, w, 3);
% normal_map(:,:,1) = reshape(nx, h, w);
% normal_map(:,:,2) = reshape(ny, h, w);
% normal_map(:,:,3) = reshape(nz, h, w);
% subplot(1,2,2);draw_normal(normal_map);title('normal by depth');

for iter=1:5
    %% estimate lighting coefficients
    t_light=tic;
    Y = makeY(nx, ny, nz);
    
    A_pixels = [ar(:); ag(:); ab(:)];
    I_pixels = [Ir(:); Ig(:); Ib(:)];
    
    if iter == 1
        second_order_weight = 0;
    else
        second_order_weight = 1;
    end
    s_weights = [ones(1, 4), ones(1, 5)*second_order_weight];
    lhs = repmat(A_pixels, 1, 9) .* [Y;Y;Y] .* repmat(s_weights, size(A_pixels,1), 1);
    rhs = I_pixels;
    
    % remove hair pixels
    lhs(hair_pixel_indices, :) = 0;
    rhs(hair_pixel_indices, :) = 0;
    
    lhs(bad_indices, :) = [];
    rhs(bad_indices, :) = [];
    
    if iter == 1
        l_lambda = 1e-3;
        l = (lhs' * lhs + l_lambda*eye(9)) \ (lhs' * rhs);
        l(5:end) = 0;
    else
        l_lambda = 1e-3;
        dl = (lhs' * lhs + l_lambda*eye(9)) \ (lhs' * rhs) - l;
        l = l + 0.01 * dl;
    end
    
    Yl = Y * l;
    
    figure(14);plot_lighting(l, false);
    
    fprintf('lithting estimation finished in %.3fs\n', toc(t_light));
    
    %% estimate albedo
    t_albedo = tic;
    w_reg = 100.0 * (0.95^(iter-1));
    A_up = sparse(1:h*w, 1:h*w, Yl(:), h*w, h*w);
    A_reg = w_reg * albedo_mat_LoG;
    
    M_reg = spdiags(ones(num_pixels, 1), 0, num_pixels, num_pixels);
    A = [A_up(good_indices_1,good_indices_1); A_reg(good_indices_1,good_indices_1)];
    
    Br = [Ir(good_indices_1); LoG_ar(good_indices_1) * w_reg];
    ar_sub = (A'*A + 0.01*M_reg) \ (A'*Br);
    Bg = [Ig(good_indices_1); LoG_ag(good_indices_1) * w_reg];
    ag_sub = (A'*A + 0.01*M_reg) \ (A'*Bg);
    Bb = [Ib(good_indices_1); LoG_ab(good_indices_1) * w_reg];
    ab_sub = (A'*A + 0.01*M_reg) \ (A'*Bb);
    
    ar(good_indices_1) = ar_sub;
    ag(good_indices_1) = ag_sub;
    ab(good_indices_1) = ab_sub;
    
    albedo_map = zeros(h, w, 3);
    albedo_map(:,:,1) = ar; albedo_map(:,:,2) = ag; albedo_map(:,:,3) = ab;
    figure(18);imshow(albedo_map);title(['albedo', num2str(iter)]);
    
    fprintf('albedo estimation finished in %.3fs\n', toc(t_albedo));    
    
    %% estimate geometry, optimize depth directly
    t_normal = tic;
    % nx = cos(theta), ny = sin(theta)*cos(phi), nz = sin(theta)*sin(phi)
    
    if false
        %dx_mat = spdiags(abs(1 ./ (C_x * x0(:))), 0, h*w, h*w);
        %dy_mat = spdiags(abs(1 ./ (C_y * y0(:))), 0, h*w, h*w);
        %norm_mat = spdiags(1./normal_norm(:), 0, h*w, h*w);

        dx_mat = spdiags(ones(h*w,1), 0, h*w, h*w);
        dy_mat = spdiags(ones(h*w,1), 0, h*w, h*w);
        norm_mat = spdiags(ones(h*w,1), 0, h*w, h*w);
        
        rhs_pixels = I_pixels(good_indices);
        rhs_z = z0(good_indices_1);
        
        lhs_1 = spdiags(ones(h*w,1)*l(1) + l(4)./normal_norm(:), 0, h*w, h*w);
        lhs_1 = lhs_1 + norm_mat * (l(2) * dx_mat * C_x + l(3) * dy_mat * C_y);
        lhs_r = spdiags(ar(:), 0, h*w, h*w) * lhs_1;
        lhs_g = spdiags(ag(:), 0, h*w, h*w) * lhs_1;
        lhs_b = spdiags(ab(:), 0, h*w, h*w) * lhs_1;
        lhs_pixels = [lhs_r; lhs_g; lhs_b];
        lhs_pixels = lhs_pixels(good_indices,:);
        lhs_z = spdiags(ones(h*w, 1), 0, h*w, h*w);
        lhs_z = lhs_z(good_indices_1, :);
        
        lhs = [lhs_pixels; ...
               lhs_z; ...
               ];
        rhs = [rhs_pixels; ...
               rhs_z; ...
               ];
        
        z1 = (lhs' * lhs) \ (lhs' * rhs);
        z1 = reshape(z1, h, w);
        dz_mask = z1 - z0;
        z = z + dz_mask;
        
        figure(16);
        subplot(1,5,1);imagesc([dz_mask;]);title('dz');axis equal;caxis([-0.05, 0.05]);
        subplot(1,5,2);imagesc([z;z0]);title('z, z0');axis equal;caxis([-0.5, 0.5]);
        Iz = I_depth;
        Iz(:,:,3) = z;
        subplot(1,5,3);plot_depth(Iz, false, true, false); title('z\_new'); axis equal;
        Iz(:,:,3) = z0;
        subplot(1,5,4);plot_depth(Iz, false, true, false); title('z\_0'); axis equal;
        
        nx = C_x * z(:) ./ abs(C_x * x0(:));
        ny = C_y * z(:) ./ abs(C_y * y0(:));
        nz = ones(h*w,1);
        
        normal_norm = sqrt(nx.^2 + ny.^2 + 1);
        nx = nx ./ normal_norm; nx(~good_indices_1) = 0;
        ny = ny ./ normal_norm; ny(~good_indices_1) = 0;
        nz = nz ./ normal_norm; nz(~good_indices_1) = 0;
        
        normal_map = zeros(h, w, 3);
        normal_map(:,:,1) = reshape(nx, h, w);
        normal_map(:,:,2) = reshape(ny, h, w);
        normal_map(:,:,3) = reshape(nz, h, w);
        normal_map = (normal_map + 1.0) * 0.5;
        subplot(1,5,5);imshow(normal_map); pause;
        
        Y = makeY(nx, ny, nz);
        Yl = Y * l;
        size(Yl)
        size(A_pixels)
        fitted_verify = A_pixels .* repmat(Yl, 3, 1);
        R_data_verify = I_pixels - A_pixels .* repmat(Yl, 3, 1);
        R_data_verify(~good_indices, :) = .5;
        residue_mask_verify = zeros(h, w);
        Rmat_verify = reshape(R_data_verify, h*w, 3);
        residue_mask_verify(:) = sum(Rmat_verify.*Rmat_verify,2);
        figure(17);
        subplot(1,4,1);imshow(reshape(I_pixels, h, w, 3));title('input');
        subplot(1,4,2);imshow(reshape(fitted_verify, h, w, 3));title('fitted verify');
        subplot(1,4,3);imshow(reshape(R_data_verify, h, w, 3)); title('residue verify');
        lighting_mask = reshape(Yl, h, w);
        subplot(1,4,4);imagesc(lighting_mask); title('lighting\_new'); axis equal; colorbar; colormap gray;
    else
        for i=1:1
            % data term
            R_data = I_pixels - A_pixels .* repmat(Yl, 3, 1);
            
            %         R_data(~good_indices, :) = .5;
            %         residue_mask = zeros(h, w);
            %         Rmat = reshape(R_data, h*w, 3);
            %         residue_mask(:) = sum(Rmat.*Rmat,2);
            %         figure;
            %         subplot(1,2,1);imshow(reshape(A_pixels .* repmat(Yl,3,1), h, w, 3));
            %         subplot(1,2,2);imshow(residue_mask); title('residue'); axis equal; pause;
            
            R_data = R_data(good_indices, :);
            fprintf('norm(R_data) = %.6f\n', norm(R_data));
            
            % Jacobians
            % Y = [1, nx, ny, nz, nx*ny, nx*nz, ny*nz, nx*nx-ny*ny, 3*nz*nz-1]
            
            dx_mat = spdiags(abs(1 ./ (C_x * x0(:))), 0, h*w, h*w);
            dy_mat = spdiags(abs(1 ./ (C_y * y0(:))), 0, h*w, h*w);
            norm_mat = spdiags(1./normal_norm(:), 0, h*w, h*w);
            norm_mat2 = spdiags(1./(normal_norm(:).*normal_norm(:)), 0, h*w, h*w);
            
            %figure;imagesc(reshape(1./normal_norm, h, w));pause;
            
            %         n_x = norm_mat * dx_mat * C_x * z(:);
            %         n_y = norm_mat * dy_mat * C_y * z(:);
            %         figure;
            %         subplot(1,2,1);imagesc(reshape(n_x, h, w)); axis equal;
            %         subplot(1,2,2);imagesc(reshape(n_y, h, w)); axis equal; pause;
            
%             dz_dx = C_x * z(:);
%             figure;plot(dz_dx(good_indices_1));
%             dz_dy = C_y * z(:);
%             figure;plot(dz_dy(good_indices_1)); pause;
            
            dR_dz = norm_mat * (l(2) * dx_mat * C_x + l(3) * dy_mat * C_y);
            dR_dz = dR_dz + norm_mat2 * sparse(1:h*w, 1:h*w, 2 * l(5) * dx_mat * dy_mat * C_x' * C_y * z(:) + 2 * l(8) * (dx_mat * dx_mat * C_x' * C_x - dy_mat * dy_mat * C_y'*C_y) * z(:));
            dR_dz = -dR_dz;
            J_data_r = spdiags(ar(:), 0, h*w, h*w) *  dR_dz;
            J_data_g = spdiags(ag(:), 0, h*w, h*w) *  dR_dz;
            J_data_b = spdiags(ab(:), 0, h*w, h*w) *  dR_dz;
            J_data = [J_data_r; J_data_g; J_data_b];
            
            J_data = J_data(good_indices,good_indices_1);
            
            %         % integrability term, this should be enforced implicitly
            %         w_int = 1.0;
            %         R_int = (C_x * C_y - C_y * C_x) * z(:);
            %         R_int = R_int(good_indices_1);
            %         R_int = R_int * w_int;
            %         norm(R_int)
            %
            %         J_int = C_x * C_y - C_y * C_x;
            %         J_int = J_int(good_indices_1, good_indices_1);
            %         J_int = J_int * w_int;
            
            % regularization term
            w_reg = 1.0;%10.0 - 2.0 * (iter - 1);
            R_reg = z(:) - z0(:);
            R_reg(discontinuous_pixels_indices,:) = R_reg(discontinuous_pixels_indices,:) * 10;
            R_reg = R_reg(good_indices_1);
            R_reg = R_reg * w_reg;
            fprintf('norm(R_reg) = %.6f\n', norm(R_reg));
            
            J_reg = spdiags(ones(h*w,1), 0, h*w, h*w);
            J_reg(discontinuous_pixels_indices,:) = J_reg(discontinuous_pixels_indices,:) * 10;
            J_reg = J_reg(good_indices_1, good_indices_1);
            J_reg = J_reg * w_reg;
            
            % regularization term 2, with Laplacian
            w_reg2 = 10.0;
            R_reg2 = mat_LoG * (z(:) - z0(:));
            R_reg2(discontinuous_pixels_indices,:) = 0;
            %b_reg(edge_pixel_indices) = 0;
            %b_reg(discontinuous_pixels_indices) = 0;
            R_reg2 = R_reg2(valid_pixel_indices) * w_reg2;  
            fprintf('norm(R_reg2) = %.6f\n', norm(R_reg2));
            
            J_reg2 = mat_LoG;
            %A_reg(edge_pixel_indices, edge_pixel_indices) = 0;
            %A_reg(discontinuous_pixels_indices, discontinuous_pixels_indices) = 0;
            J_reg2 = J_reg2(valid_pixel_indices, valid_pixel_indices) * w_reg2;
            
            % curvature term
            w_cur = 10.0;
            R_cur = [C_x * C_x * z(:); C_y * C_y * z(:); 2 * C_x * C_y * z(:)];
            R_cur(discontinuous_pixels_indices,:) = 0;
            R_cur = R_cur(good_indices,:);
            R_cur(abs(R_cur)>10) = 0 ;
            %figure;histogram(R_cur);pause;
            fprintf('norm(R_cur) = %.6f\n', norm(R_cur));
            J_cur = [C_x * C_x; C_y * C_y; 2 * C_x * C_y];
            J_cur = J_cur(good_indices, good_indices_1);                        
            J_cur = J_cur * w_cur;
            
            % solve it
            %         fprintf('solving for normal ...\n');
            %w_lambda = 100.0 - 20.0 * (iter - 1);%1.0 - 0.25 * (iter-1);
            w_lambda = 10.0;
            M_reg = spdiags(ones(size(J_data, 2), 1), 0, size(J_data, 2), size(J_data, 2));
            
            J = [J_data; ...
                %J_int; ...
                J_reg; ...
                J_reg2; ...
                J_cur;
                ];
            R = [R_data; ...
                %R_int; ...
                R_reg; ...
                R_reg2; ...
                R_cur
                ];
            
            JTJ = J' * J;
            JTR = J' * R;
            
            fprintf('cond(JTJ) = %g', condest(JTJ));
            
            dz = (JTJ + w_lambda * M_reg) \ JTR;
            
            %         [Rcdf, xcenter] = hist(R); figure;plot(xcenter, Rcdf);title('R');
            %         figure;plot(dx);title('dx');
            %         [JJJR, xcenter] = hist(dx);figure;plot(xcenter, JJJR);title('hist dx');
            
            figure(15);histogram(dz);
            dz_alpha = 0.25;
            %dz(abs(dz)>0.1) = 0;
            z(good_indices_1) = z(good_indices_1) - dz(1:num_pixels) * dz_alpha;
            
            %         % copy the edge depths to neighboring pixels
            %         se = strel('disk',2);
            %         z = imdilate(z, se);
            
            %z = z - reshape(dz, h, w) * dz_alpha;
            %         fprintf('done.\n');
            
            dz_mask = zeros(h,w); dz_mask(good_indices_1) = dz(1:num_pixels);
            %dz_mask = reshape(dz, h, w);
            figure(16);
            subplot(1,5,1);imagesc([dz_mask;]);title('dz, z, z0');axis equal;caxis([-0.05, 0.05]);
            subplot(1,5,2);imagesc([z;z0]);title('dz, z, z0');axis equal;caxis([-0.5, 0.5]);
            Iz = I_depth;
            Iz(:,:,3) = z;
            subplot(1,5,3);plot_depth(Iz, false, true, false); title('z\_new'); axis equal;
            Iz(:,:,3) = z0;
            subplot(1,5,4);plot_depth(Iz, false, true, false); title('z\_0'); axis equal;
            
            nx = C_x * z(:) ./ abs(C_x * x0(:));
            ny = C_y * z(:) ./ abs(C_y * y0(:));
            nz = ones(h*w,1);
            
            normal_norm = sqrt(nx.^2 + ny.^2 + 1);
            nx = nx ./ normal_norm; nx(~good_indices_1) = 0;
            ny = ny ./ normal_norm; ny(~good_indices_1) = 0;
            nz = nz ./ normal_norm; nz(~good_indices_1) = 0;
            
            normal_map = zeros(h, w, 3);
            normal_map(:,:,1) = reshape(nx, h, w);
            normal_map(:,:,2) = reshape(ny, h, w);
            normal_map(:,:,3) = reshape(nz, h, w);
            normal_map = (normal_map + 1.0) * 0.5;
            subplot(1,5,5);imshow(normal_map);
            
            Y = makeY(nx, ny, nz);
            Yl = Y * l;

            fitted_verify = A_pixels .* repmat(Yl, 3, 1);
            R_data_verify = I_pixels - A_pixels .* repmat(Yl, 3, 1);
            R_data_verify(~good_indices, :) = .5;
            residue_mask_verify = zeros(h, w);
            Rmat_verify = reshape(R_data_verify, h*w, 3);
            residue_mask_verify(:) = sum(Rmat_verify.*Rmat_verify,2);
            figure(17);
            subplot(1,4,1);imshow(reshape(I_pixels, h, w, 3));title('input');
            subplot(1,4,2);imshow(reshape(fitted_verify, h, w, 3));title('fitted verify');
            subplot(1,4,3);imshow(reshape(R_data_verify, h, w, 3)); title(sprintf('residue verify = %.6f', sum(residue_mask_verify(good_indices_1))));
            lighting_mask = reshape(Yl, h, w);
            subplot(1,4,4);imagesc(lighting_mask); title('lighting\_new'); axis equal; colorbar; colormap gray;
        end
    end
    fprintf('normal estimation finished in %.3fs\n', toc(t_normal));
end

if false
    hfig = figure;
    subplot(1, 4, 1); imshow(I_input); title('input');
    subplot(1, 4, 2); imshow(albedo_map); title('albedo');
    subplot(1, 4, 3); imshow(I_normal); title('normal');
    subplot(1, 4, 4); imshow(normal_map); title('refined normal');
    set(hfig, 'Position', [0 0 1200 480])
    pause
    
    return
end

%% recover depth
[dh, dw, ~] = size(I_depth);
final_normal_map = imresize(normal_map*2.0-1.0, [dh, dw]);

depth0 = I_depth(:,:,3);

valid_depth_points = depth0(:)>-1e5;
edge_pixel_indices = find_depth_edge(depth0, 1);
valid_depth_points(edge_pixel_indices) = 0;

I_depth_final = I_depth;

% update the depth
depth0(valid_depth_points) = z(valid_depth_points);
depth0(~valid_depth_points) = -1e6;
I_depth_final(:,:,3) = depth0;

plot_depth(I_depth_final, true);

% zumz = spdiags(abs(1.0 ./ (Cy * y0(:))), 0, dh*dw, dh*dw) * Cy * depth0(:); zumz = reshape(zumz, dh, dw);
% zumz(~valid_depth_points) = 0;
% figure;imagesc(zumz); title('zumz - final'); axis equal; caxis([-0.5, 0.5]);
% zmzl = spdiags(abs(1.0 ./ (Cx * x0(:))), 0, dh*dw, dh*dw) * Cx * depth0(:); zmzl = reshape(zmzl, dh, dw);
% zmzl(~valid_depth_points) = 0;
% figure;imagesc(zmzl); title('zmzl - final'); axis equal; caxis([-0.5, 0.5]);

hfig = figure;
subplot(1, 7, 1); imshow(I_input); title('input');
subplot(1, 7, 2); imshow(albedo_map); title('albedo');
subplot(1, 7, 3); plot_lighting(l, false); title('lighting');
subplot(1, 7, 4); imshow(I_normal); title('normal');
subplot(1, 7, 5); imshow(normal_map); title('refined normal');
subplot(1, 7, 6); plot_depth(I_depth, false, true); title('init depth');
subplot(1, 7, 7); plot_depth(I_depth_final, false, true); title('refined depth');
set(hfig, 'Position', [0 0 1200 480])

% save all output images
imwrite(albedo_map, fullfile(options.path, 'SFS', sprintf('optimized_albedo_%d.png', options.idx)));
imwrite(normal_map, fullfile(options.path, 'SFS', sprintf('optimized_normal_%d.png', options.idx)));
plot_lighting(l, true, fullfile(options.path, 'SFS', sprintf('optimized_lighting_%d.png', options.idx)));
save_depth_map(fullfile(options.path, 'SFS', sprintf('optimized_depth_map_%d.bin', options.idx)), I_depth_final);
save_point_cloud(fullfile(options.path, 'SFS', sprintf('optimized_point_cloud_%d.txt', options.idx)), I_depth_final);
plot_depth(I_depth, false, true, true, fullfile(options.path, 'SFS', sprintf('depth_mesh_%d.png', options.idx)));
plot_depth(I_depth_final, false, true, true, fullfile(options.path, 'SFS', sprintf('optimized_depth_mesh_%d.png', options.idx)));

%pause;

end

function Y = makeY(nx, ny, nz)
Y = [ones(size(nx)), nx, ny, nz, nx.*ny, nx.*nz, ny.*nz, nx.*nx-ny.*ny, 3*nz.*nz-1];
end

function plot_mask(title_str, pixel_indices, h, w)
mask = zeros(h, w);
mask(pixel_indices) = 1;
figure;imshow(mask); title(title_str);
end
