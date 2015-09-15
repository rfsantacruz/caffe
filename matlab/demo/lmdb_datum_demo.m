% Example of using lmdb and caffe protobuf (datum) in matlab. 
%
% by Jiayu, July 1, 2015. 
%
% NOTE 1. start matlab with a specified libtiff.5.dylib.
%    DYLD_INSERT_LIBRARIES=/usr/local/lib/libtiff.5.dylib /Applications/MATLAB_R2012b.app/bin/matlab &
%
%      2. install matlab-lmdb 
%   https://github.com/illidanlab/matlab-lmdb
%
%      3. the image num (the first input_num) in the model file should set to 1. 
%         will fix later. 

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

addpath /home/rfsantacruz/Documents/Softwares/matlab-lmdb-master  % change to your matlab-lmdb path

cur_director = pwd;
net_model    = strcat(cur_director, '/../../examples/mnist/lenet.prototxt');
net_weights  = strcat(cur_director, '/../../examples/mnist/lenet_iter_10000.caffemodel');
db_path      = strcat(cur_director, '/../../examples/mnist/mnist_test_lmdb');
use_gpu = 0;
phase = 'test';


% create caffe net instance
caffe.set_mode_cpu();
net = caffe.Net(net_model, net_weights, phase);

% load an existing lmdb database (crated using the shell in example). 
database = lmdb.DB(db_path, 'RDONLY', true, 'NOLOCK', true);
cursor = database.cursor('RDONLY', true);

max_count = 10; % maximum test cases

count = 0;      
correctNum = 0;
while cursor.next()
  key = cursor.key;
  value = cursor.value;
  
  % transform datum. 
  [image, label] = caffe.fromDatum(value);
  
  % prepare image
  data = single(image);
  data = permute(data, [2,1,3]);
  data = cat(4,data,zeros(28,28,1,63));
  
  % generate prediction 
  scores = net.forward({data});
  scores{1} = scores{1}(:,1);
  predict_class = find(scores{1}==1) - 1; % shift 1
  
  
  fprintf('[%u] Class %u predicted as %u \n', count+1, label, predict_class)
  
  if(predict_class == label)
      correctNum = correctNum + 1;
  end
  
  count = count + 1;
  if (count >= max_count)
      break;
  end
end

fprintf('Correctly classified %d images out of %d ( %d percent)\n', correctNum, count, correctNum/count * 100)

clear cursor;

%test toDatum and fromDatum
fprintf('Testing Conversion between datum and (image, label)...\n')
gt_img = rand(28,28,3,64, 'single'); gt_lab = randi(100);
datum = caffe.toDatum(gt_img, gt_lab);
[t_img, t_lab] = caffe.fromDatum(datum); t_img = reshape(t_img,[28,28,3,64]);
error = sum(gt_img(:) ~= t_img(:)) + (gt_lab ~= t_lab);
fprintf('Conversion done with %d errors found\n', error);



