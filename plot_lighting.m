function plot_lighting(l, saveit, filename)
if saveit
    fig = figure;
    step_factor=2.0;
else
    step_factor=1.0;
end
[theta_vis, phi_vis] = ndgrid(-pi:pi/(32*step_factor):0.*pi, 0:2*pi/(64*step_factor):2*pi);
x_vis = cos(theta_vis);
y_vis = sin(theta_vis) .* cos(phi_vis);
z_vis = sin(theta_vis) .* sin(phi_vis);
n_vis = [x_vis(:), y_vis(:), z_vis(:)];
Y_vis = makeY(x_vis(:), y_vis(:), z_vis(:));
Yl_vis = Y_vis * l;

%n_vis = n_vis .* repmat(Yl_vis, 1, 3);

surf(reshape(n_vis(:,1), size(x_vis)), ...
     reshape(n_vis(:,2), size(y_vis)), ...
     reshape(n_vis(:,3), size(z_vis)), ...
     reshape(Yl_vis, size(z_vis)), ...
     'edgecolor', 'none');
%title(sprintf('lighting %d', iter)); 
axis equal; colorbar;
xlabel('x');ylabel('y');zlabel('z');
view([0 90]);
if saveit
    saveas(fig, filename);
    close(fig);
end
end

function Y = makeY(nx, ny, nz)
Y = [ones(size(nx)), nx, ny, nz, nx.*ny, nx.*nz, ny.*nz, nx.*nx-ny.*ny, 3*nz.*nz-1];
end