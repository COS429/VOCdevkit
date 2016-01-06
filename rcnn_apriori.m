% rcnn
% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);
% initialize VOC options
VOCinit;
% load specified set
set = VOCopts.testset;

%% Save regions as distinct images in tensor format
tic;
addpath(genpath('/home/aseff/marvin/'));

% load specified set
tic;
gtids=textread(sprintf(VOCopts.imgsetpath,set),'%s');
if(false)
    for i=1:length(gtids)
        % display progress
        if toc>1
            fprintf('load annotations: %d/%d\n',i,length(gtids));
            drawnow;
            tic;
        end
        
        % read annotation
        recs(i)=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    end
else
    load('recs_test.mat');
end

numBB = 10;
imgsize = 227;
data = zeros(227, 227, 3, length(gtids)*numBB, 'single');
labels = zeros(1,1,1,length(gtids)*numBB, 'single');

% A priori boxes
load('kmeans_boxes.mat');

tic;
count = 1;
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('load actual: %d/%d\n',i,length(gtids));
        drawnow;
        tic;
    end
    
    % load image
    img=imread(sprintf(VOCopts.imgpath,gtids{i}));
    % Adjust boxes for previous scaling
    X = recs(i).size.width; Y = recs(i).size.height;
    boxes = kmeans_boxes .* repmat([X, Y, X, Y], size(kmeans_boxes,1), 1);
    boxes = round(boxes);
    
    % Iterate through boxes and crop specified regions
    box_labels = zeros(size(boxes,1), 1);
    for cr = 1:size(boxes,1)
        % Assess intersection over union with each ground truth object
        for ob = 1:length(recs(i).objects)
            gt_bbox = recs(i).objects(ob).bbox;
            if(IoU(boxes(cr,:), gt_bbox) > 0.5)
                % Leave label as zero (background) unless a match is found
                box_labels(cr) = find(ismember(VOCopts.classes, recs(i).objects(ob).class));
                break;
            end
        end
    end
    
    for cr = 1:size(boxes,1)
        cropped_im = img(boxes(cr,2):boxes(cr,4), boxes(cr,1):boxes(cr,3), :);
        cropped_im = imresize(cropped_im, [imgsize imgsize]);
        data(:,:,:,count) = cropped_im;
        labels(:,:,:,count) = box_labels(cr);
        count = count + 1;
    end
end



%% Save tensors
saveDir = '/data/aseff/rcnn';

tensor.type = 'half';
tensor.sizeof = 2;
tensor.name = 'train_regions';
tensor.value = single(data);
tensor.dim = 4;
writeTensors(sprintf('%s/train_regions.tensor', saveDir), tensor);

tensor.name = 'train_labels';
tensor.value = single(labels);
writeTensors(sprintf('%s/train_labels.tensor', saveDir), tensor);


%% Run AlexNet on extracted regions
% cd('/home/aseff/marvin');
% setenv('LD_LIBRARY_PATH', '/usr/local/cuda/lib64/:/usr/local/cudnn/v4rc/lib64');
% setenv('PATH', '/usr/local/cuda/bin/');
% system('./marvin test rcnn/alexnet_imagenet.json models/alexnet_imagenet/alexnet_imagenet_half.marvin loss rcnn/prob_file.tensor');
% 
% 
