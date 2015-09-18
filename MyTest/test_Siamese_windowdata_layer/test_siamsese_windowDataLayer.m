clear all;

number = '981323206';
img_path = sprintf('/home/rfsantacruz/Documents/Project/caffe/dump/%s_data.txt',number);
info_path = sprintf('/home/rfsantacruz/Documents/Project/caffe/dump/%s_info.txt',number);

%read info
f= fopen(info_path);
Pair.id = fgetl(f);
Pair.A.img_path = fgetl(f);
Pair.A.window = [str2num(fgetl(f)), str2num(fgetl(f)), str2num(fgetl(f)), str2num(fgetl(f))];
Pair.A.mirror = str2num(fgetl(f));

Pair.B.img_path = fgetl(f);
Pair.B.window = [str2num(fgetl(f)), str2num(fgetl(f)), str2num(fgetl(f)), str2num(fgetl(f))];
Pair.B.mirror = str2num(fgetl(f));

Pair.sim = str2num(fgetl(f));
Pair.isfg = str2num(fgetl(f));
fclose(f);

%crop original image
Pair.A.img = imcrop(imread(Pair.A.img_path), Pair.A.window - [0 0 Pair.A.window(1) Pair.A.window(2)]);
Pair.B.img = imcrop(imread(Pair.B.img_path), Pair.B.window - [0 0 Pair.B.window(1) Pair.B.window(2)]);


%read data layer
f= fopen(img_path);
M = fread(f,'single');
M = reshape(M, [227, 227, 6]); 

Pair.A.data =  permute(M(:,:,1:3), [2, 1, 3]);
if(Pair.A.mirror)
    Pair.A.data = flip(Pair.A.data,2);
end

Pair.B.data = permute(M(:,:,4:6), [2, 1, 3]);
if(Pair.B.mirror)
    Pair.B.data = flip(Pair.B.data,2);
end

fclose(f); clear M;

%show original images and data layer attentio to the mirror
subplot(2,2,1), imshow(Pair.A.img);
subplot(2,2,2), imshow(Pair.A.data);
subplot(2,2,3), imshow(Pair.B.img);
subplot(2,2,4), imshow(Pair.B.data);
