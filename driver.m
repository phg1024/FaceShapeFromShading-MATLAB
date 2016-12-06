close all;

person = 'Oprah_Winfrey';
%person = 'Hillary_Clinton';
%person = 'Benedict_Cumberbatch';
%person = 'Donald_Trump';
%person = 'George_W_Bush';
%person = 'Zhang_Ziyi';
%person = 'Andy_Lau';
%person = 'Jackie_Chan';
path = sprintf('/home/phg/Storage/Data/InternetRecon2/%s/crop', person);

person = 'yaoming';
path = sprintf('/home/phg/Storage/Data/InternetRecon0/%s/crop', person);

% LoG kernel and LoG matrix
[LoG, mat_LoG] = LoGMatrix(2, 250, 250, 1.0);
[albedo_LoG, albedo_mat_LoG] = LoGMatrix(2, 250, 250, 0.5);

options.LoG = LoG;
options.mat_LoG = mat_LoG;
options.albedo_LoG = albedo_LoG;
options.albedo_mat_LoG = albedo_mat_LoG;
options.path = path;

options.silent = true;

all_images = read_settings(fullfile(path, 'settings.txt'));

for i=1:length(all_images)
input_image = fullfile(path, all_images{i});
[~, basename, ~] = fileparts(all_images{i});
albedo_image = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', i-1));
normal_image = fullfile(path, 'SFS', sprintf('normal%d.png', i-1));
mask_image = fullfile(path, 'masked', sprintf('mask%s.png', basename));
depth_map = fullfile(path, 'SFS', sprintf('depth_map%d.bin', i-1));

options.idx = i-1;

%[h, w, ~] = size(imread(albedo_image));
%[LoG, mat_LoG] = LoGMatrix(2, h, w, 1.0);

tic;
refined_normal_map = SFS(input_image, albedo_image, normal_image, depth_map, mask_image, options);
fprintf('image %d finished in %.3fs\n', i, toc);

pause(2);
end

% TODO
% restore image scaling in ioutilites
% regenerate large scale
% prepare for SFS
% figure out how to generate mesh from SFS point cloud