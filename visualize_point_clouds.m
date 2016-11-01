idx = 32;
database = '~/Storage/Data/Internet0/%s/crop/';
person = 'yaoming';

%path='~/Storage/Data/InternetRecon2/Zhang_Ziyi/crop/';
path = sprintf(database, person);

A = importdata(fullfile(path, 'SFS', sprintf('optimized_point_cloud_%d.txt', idx)));
B = importdata(fullfile(path, 'SFS', sprintf('point_cloud%d.txt', idx)));
figure; hold on;
plot3(A(:,1), A(:,2), A(:,3), 'b.');
plot3(B(:,1), B(:,2), B(:,3), 'r.');
axis equal;
