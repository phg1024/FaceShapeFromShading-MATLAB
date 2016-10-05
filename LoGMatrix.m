function [kernel, m] = LoGMatrix(k, h, w, sigma)
kernel_shape = [2*k+1, 2*k+1];
kernel = fspecial('log', kernel_shape, sigma);

num_pixels = h * w;
rows = cell(num_pixels, 1);
cols = cell(num_pixels, 1);
elem = cell(num_pixels, 1);

for j=1:w
    for i=1:h
        id = (j-1)*h+i;
        
        hr = 1:2*k+1;
        hc = 1:2*k+1;
        
        r = i-k:i+k;
        c = j-k:j+k;
        
        % remove out of range pixels in the kernel
        hr(r<=0) = []; hr(r>h) = [];        
        hc(c<=0) = []; hc(c>w) = [];
        
        % remove out of range pixels in the image matrix
        r(r<=0) = []; r(r>h) = [];
        c(c<=0) = []; c(c>w) = [];
        
        [r_final, c_final] = meshgrid(r, c);
        [hr_final, hc_final] = meshgrid(hr, hc);
        
        neighbor_ids = sub2ind([h, w], r_final(:), c_final(:));
        
        elem{id} = kernel(sub2ind(kernel_shape, hr_final(:), hc_final(:)));
        cols{id} = neighbor_ids(:);
        rows{id} = ones(size(neighbor_ids)) * id;
    end
end

all_rows = cell2mat(rows);
all_cols = cell2mat(cols);
all_elem = cell2mat(elem);

m = sparse(all_rows, all_cols, all_elem, h*w, h*w);
end