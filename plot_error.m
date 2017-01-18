function plot_error(Idiff, msk, saveit, filename)

I = create_diff_image(Idiff, msk, 0, 0.25);

if saveit
    imwrite(I, filename);
end

end
