function joint_plot(varargin)
I = [];
for i=1:length(varargin)
    [~,~,c] = size(varargin{i});
    if c == 1
        I = [I, repmat(varargin{i}, [1, 1, 3])];
    else
        I = [I, varargin{i}];
    end
end
imshow(I);
end