% bb_test
% change this path if you install the VOC code elsewhere
addpath(genpath('/home/aseff/COS429')); % contains copy of marvin and VOCdevkit

% initialize VOC options
VOCinit; % set path to data within VOCinit
train = false;
% load specified set
if(train)
    set = VOCopts.trainset;
else
    set = VOCopts.testset;
end

%% Set up selective search

% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation.
minSize = k;
sigma = 0.8;

%% Load image info

% load specified set
tic;
gtids=textread(sprintf(VOCopts.imgsetpath,set),'%s');
if(false) % pre-saved recs already
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
    if(train)
        load('recs_train.mat')
    else
        load('recs_test.mat');
    end
end

%% Prepare images, bounding boxes, and labels

imgsize = 227;
data = zeros(227, 227, 3, length(gtids), 'single');
bb = []; % Not pre-allocated (B x 5)
labels = []; % Not pre-allocated (no longer 4-dimensional) (B x 1)

tic;
count = 1;
limit = 500;
for i=1:limit %length(gtids)
    % display progress
    if toc>1
        fprintf('load image: %d/%d\n',i,length(gtids));
        drawnow;
        tic;
    end
    
    % load image
    img=imread(sprintf(VOCopts.imgpath,gtids{i}));
    
    % Perform Selective Search
    boxes = Image2HierarchicalGrouping(img, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    boxes = boxes(:,[2 1 4 3]); % to match pascal [xmin, ymin, xmax, ymax] format
    
    % Iterate through boxes and assing labels
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
    
    % Adjust image size while preserving aspect ratio
    X = recs(i).size.width; Y = recs(i).size.height;
    if(X < Y)
        img = imresize(img, [227 NaN]);
    else
        img = imresize(img, [NaN 227]);
    end
    x_scale = size(img, 2)/X;
    y_scale = size(img, 1)/Y;
    X = size(img, 2); Y = size(img, 1);
    % Zero pad
    img = padarray(img, [227-Y, 227-X], 'post');
    
    % Scale bounding boxes accordingly and round
    boxes = boxes .* repmat([x_scale, y_scale, x_scale, y_scale], size(boxes,1), 1);
    boxes = boxes - 1; % for zero-indexing
    boxes = round(boxes);
    inds = find(boxes < 0);
    boxes(inds) = 0;
  
    % Add global image id to bounding boxes
    boxes = [(count-1)*ones(size(boxes,1),1), boxes];
    
    % Append data to full tensors
    data(:,:,:,count) = img;
    count = count + 1;
    bb = [bb; boxes];
    labels = [labels; box_labels];
end



%% Save tensors
saveDir = '/data/aseff/pascal';
if(train)
    pref = 'train';
else
    pref = 'test';
end

% Save images
tensor.type = 'half';
tensor.sizeof = 2;
tensor.name = [pref '_images'];
tensor.value = single(data(:,:,:,1:count-1));
tensor.dim = 4;
writeTensors(sprintf('%s/%s.tensor', saveDir, tensor.name), tensor);

% Save mean image if training
if(train)
    tensor.type = 'half';
    tensor.sizeof = 2;
    tensor.name = 'train_mean';
    tensor.value = single(mean(data(:,:,:,1:count-1), 4));
    tensor.dim = 3;
    writeTensors(sprintf('%s/%s.tensor', saveDir, tensor.name), tensor);
end

% Save bounding boxes
% Modify to match ROIPooling expected input
bb = bb(:, [1 3 5 2 4]); % Now [ymin, ymax, xmin, xmax]
newBBs = zeros(1,1,5,size(bb,1));
newBBs(1,1,:,:) = bb';
bb = newBBs;
tensor.type = 'half';
tensor.sizeof = 2;
tensor.name = [pref '_bboxes'];
tensor.value = single(bb);
tensor.dim = 4;
writeTensors(sprintf('%s/%s.tensor', saveDir, tensor.name), tensor);

% Save bounding box labels
newLabels = zeros(1,1,1,size(labels,1));
newLabels(1,1,1,:) = labels;
labels = newLabels;
tensor.type = 'half';
tensor.sizeof = 2;
tensor.name = [pref '_labels'];
tensor.value = single(labels);
tensor.dim = 4;
writeTensors(sprintf('%s/%s.tensor', saveDir, tensor.name), tensor);
