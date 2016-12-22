function indices = read_indices(filename)
fid = fopen(filename, 'r');

indices = fscanf(fid, '%d');

fclose(fid);
end