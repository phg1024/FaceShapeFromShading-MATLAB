function plot_error(Idiff, msk, saveit, filename)

I = create_diff_image(Idiff, msk, 0, 0.25);

if saveit
    imwrite(I, filename);
    imwrite(imgaussfilt(Idiff, 1) < 0.025, [filename, '_mask.jpg']);
end

end
