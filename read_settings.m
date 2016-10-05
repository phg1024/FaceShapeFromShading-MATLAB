function all_lines = read_settings(filename)

fid = fopen(filename, 'r');

line_id = 1;
all_lines = {};
tline = fgets(fid);
while ischar(tline)
    disp(tline)
    parts = strsplit(tline);
    all_lines{line_id} = parts{1};
    line_id = line_id + 1;
    tline = fgets(fid);
end

fclose(fid);
end