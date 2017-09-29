close all;

person = 'Oprah_Winfrey';
person = 'Hillary_Clinton';
%person = 'Benedict_Cumberbatch';
person = 'Donald_Trump';
%person = 'George_W_Bush';
%person = 'Zhang_Ziyi';
%person = 'Andy_Lau';
%person = 'Jackie_Chan';
path = sprintf('/home/phg/FastStorage/Data/InternetRecon3/%s', person);

person = 'yaoming';
person = 'Andy_Lau';
%person = 'Jennifer_Aniston';
path = sprintf('/home/phg/Data/InternetRecon0/%s', person);

% LoG kernel and LoG matrix
if ~exist('LoG') || ~exist('mat_LoG')
    [LoG, mat_LoG] = LoGMatrix(2, 250, 250, 1.0);
end
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
    [~, basename, ~] = fileparts(all_images{i})
    
    %if ~strcmp(basename, '75')
    %    continue;
    %end
    
    % create albedo image    
    addpath ~/Codes/Reference/L1Flattening/matlab/    
    addpath ~/Codes/Reference/L1Flattening/src/
    albedo_image = fullfile(path, 'albedos', sprintf('albedo_flattened_%d.png', i-1));
    
    if exist(albedo_image, 'file') == 2
        
    else
        flattened_image = l1flattening(imread(input_image), struct());    
        imwrite(flattened_image, albedo_image);
    end
    
    %albedo_image = fullfile(path, 'SFS', sprintf('albedo_transferred_%d.png', i-1));
    normal_image = fullfile(path, 'SFS', sprintf('normal%d.png', i-1));
    mask_image = fullfile(path, 'masked', sprintf('mask%s.png', basename));
    depth_map = fullfile(path, 'SFS', sprintf('depth_map%d.bin', i-1));

    options_i = options;
    options_i.idx = i-1;

    %[h, w, ~] = size(imread(albedo_image));
    %[LoG, mat_LoG] = LoGMatrix(2, h, w, 1.0);

    tic;
    refined_normal_map = SFS(input_image, albedo_image, normal_image, depth_map, mask_image, options_i);
    fprintf('image %d finished in %.3fs\n', i, toc);

    pause;
end

% TODO
% restore image scaling in ioutilites
% regenerate large scale
% prepare for SFS
% figure out how to generate mesh from SFS point cloud
