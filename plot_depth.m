function plot_depth(depth)
x = depth(:,:,1);
y = depth(:,:,2);
z = depth(:,:,3);

invalid_points = z<-1e3;

x(invalid_points) = [];
y(invalid_points) = [];
z(invalid_points) = [];

figure; 
trisurf(delaunay(x, y), x, y, z, 'edgecolor', 'none', 'facecolor', [0.5, 0.5, 0.5]); material dull;
light('Position',[-2 3 5],'Style','local', 'Color', [0.75, 0.75, 0.75])
light('Position',[2 3 5],'Style','local', 'Color', [0.75, 0.75, 0.75])
xlabel('x'); ylabel('y'); zlabel('z');
axis equal; view([0 90]);
end