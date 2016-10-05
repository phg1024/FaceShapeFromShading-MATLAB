close all;

path = '/home/phg/Storage/Data/InternetRecon2/Zhang_Ziyi/crop';

all_images = read_settings(fullfile(path, 'settings.txt'));

% LoG kernel and LoG matrix
[LoG, mat_LoG] = LoGMatrix(2, 256, 256, 1.0);

for i=1:length(all_images)
input_image = fullfile(path, all_images{i});

albedo_image = fullfile(path, 'SFS', sprintf('albedo%d.png', i-1));
normal_image = fullfile(path, 'SFS', sprintf('normal%d.png', i-1));

tic;
refined_normal_map = SFS(input_image, albedo_image, normal_image, LoG, mat_LoG);
toc;

pause(2);
end