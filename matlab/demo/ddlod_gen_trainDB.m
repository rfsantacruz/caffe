function ddlod_gen_trainDB(imdb, roidb, pairs, out_dir)
%function generate text file in the following format
%   # FULL_IMAGES idx (n times)
%       path
%       channels
%       height
%       widht
%       num_bbs
%       bbidx class overlap x1 y1 x2 y2
%       .....num_bbs times
%
%   # PAIRS NUM_pairs (1 time)
%       pairidx imidx1 bbidx1 imidx2 bbidx2 sim
%       ....NUM_pairs times



window_file = sprintf('%s/window_file_%s.txt', ...
    out_dir, imdb.name);
fid = fopen(window_file, 'wt');
channels = 3; % three channel images

%%generate Image section
timer = tic;
for i = 1:length(imdb.image_ids)
  fprintf('Writing FULL_IMAGES section file: %d/%d in %d seconds\n', i, length(imdb.image_ids), toc(timer));
  img_path = imdb.image_at(i);
  roi = roidb.rois(i);
  num_boxes = size(roi.boxes, 1);
  fprintf(fid, '# FULL_IMAGES %d\n', i-1);
  fprintf(fid, '%s\n', img_path);
  fprintf(fid, '%d\n%d\n%d\n', ...
      channels, ...
      imdb.sizes(i, 1), ...
      imdb.sizes(i, 2));
  fprintf(fid, '%d\n', num_boxes);
  
  %write bouding boxes
  for j = 1:num_boxes
    label = roi.class(j); ov = roi.overlap(j, label) ;
    
    % bbs from 1-based index to 0-based index
    bbox = roi.boxes(j,:)-1;
    
    fprintf(fid, '%d %d %.3f %d %d %d %d\n', ...
        j-1, label, ov, bbox(1), bbox(2), bbox(3), bbox(4));
  end
end

num_pairs = size(pairs,1); timer = tic;
pairs(:,1:4) = pairs(:,1:4) - 1; %pairs from 1-based index to 0-based index
fprintf(fid, '# PAIRS %d\n', num_pairs);
for p = 1:num_pairs;    
    fprintf(fid, '%d %d %d %d %d %d\n', p-1, pairs(p,1), pairs(p,2), pairs(p,3), pairs(p,4), pairs(p,5));
    
    if(mod(p,100) == 0)
        fprintf('Writing DB pair section: %d/%d in %d seconds\n', p, num_pairs, toc(timer));
    end
end

fclose(fid);
