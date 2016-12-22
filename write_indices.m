function write_indices(filename, indices)
fid = fopen(filename, 'w');

fprintf(fid, '%d\n', indices);

fclose(fid);
end