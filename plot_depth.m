function plot_depth(depth, new_figure, surface)
x = depth(:,:,1);
y = depth(:,:,2);
z = depth(:,:,3);

invalid_points = z<-1e1;

x(invalid_points) = [];
y(invalid_points) = [];
z(invalid_points) = [];

if nargin < 2
    new_figure=false;
end

if new_figure
    figure; 
end

if nargin < 3
    surface = true;
end

if surface
    trisurf(delaunay(x, y), x, y, z, 'edgecolor', 'none', 'facecolor', [0.5, 0.5, 0.5]); material dull;
    light('Position',[-2 3 5],'Style','local', 'Color', [0.75, 0.75, 0.75])
    light('Position',[2 3 5],'Style','local', 'Color', [0.75, 0.75, 0.75])
else
    plot3(x, y, z, 'b.');
end
xlabel('x'); ylabel('y'); zlabel('z');
axis equal; view([0 90]);
end