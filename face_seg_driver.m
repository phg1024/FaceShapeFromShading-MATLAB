%% driver for face segmentation

clear all;
close all;

person = 'Oprah_Winfrey';
person = 'Hillary_Clinton';
person = 'Donald_Trump';
person = 'George_W_Bush';
person = 'Zhang_Ziyi';
person = 'Jackie_Chan';
person = 'Andy_Lau';
path = sprintf('/home/phg/Data/InternetRecon3/%s', person);

%person = 'yaoming';
%path = sprintf('/home/phg/Data/InternetRecon0/%s/crop', person);

%parpool('8workers', 8);

tic;
face_seg(path);
toc;