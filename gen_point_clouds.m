close all;

database_path = '/home/phg/Storage/Data/InternetRecon2/%s/crop';
persons = {'Andy_Lau',...
    'Benedict_Cumberbatch',...
    'Bruce_Willis',...
    'Donald_Trump',...
    'George_W_Bush',...
    'Hillary_Clinton',...
    'Oprah_Winfrey',...
    'Zhang_Ziyi'};
persons = {'Andy_Lau'};

%database_path = '/home/phg/Storage/Data/InternetRecon0/%s/crop';
%persons = {'yaoming'};

for j=1:length(persons)
    person=persons{j};
    path = sprintf(database_path, person);
    
    all_images = read_settings(fullfile(path, 'settings.txt'));
    
    for i=1:length(all_images)
        depth_map = fullfile(path, 'SFS', sprintf('optimized_depth_map_%d.bin', i-1));
        I_depth = load_depth_map(depth_map, [250, 250]);
        save_point_cloud(fullfile(path, 'SFS', sprintf('optimized_point_cloud_%d.txt', i-1)), I_depth);
    end
end