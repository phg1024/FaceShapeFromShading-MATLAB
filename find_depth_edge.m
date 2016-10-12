function edge_pixels = find_depth_edge(I, wsize)

edge_pixels = [];
[h, w, ~] = size(I);

if nargin < 2
    wsize = 2;
end

edge_pixels = [];
for j=1:w
    for i=1:h
        id=(j-1)*h+i;
        
        x = j-wsize:j+wsize;
        y = i-wsize:i+wsize;
        
        x(x<1) = []; x(x>w) = [];
        y(y<1) = []; y(y>h) = [];
        
        [xx, yy] = meshgrid(x, y);
        ids = sub2ind([h, w], yy(:), xx(:));
        
        if all(I(ids) > -1e5)
            ;
        else
            edge_pixels = [edge_pixels, id];
        end
    end
end

end