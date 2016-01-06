% rcnn
% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);
% initialize VOC options
VOCinit;
set = VOCopts.trainset;
% load specified set

%% Set up selective search
addpath(genpath('/home/aseff/VOCdevkit'));

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

%% Save regions as distinct images in tensor format
tic;
addpath(genpath('/home/aseff/marvin/'));

% load specified set
tic;
gtids=textread(sprintf(VOCopts.imgsetpath,set),'%s');
for i=1:length(gtids) %-5500
    % display progress
    if toc>1
        fprintf('load annotations: %d/%d\n',i,length(gtids));
        drawnow;
        tic;
    end
    
    % read annotation
    recs(i)=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
end

numBB = 10;
imgsize = 227;
data = zeros(227, 227, 3, length(gtids)*numBB, 'single');
labels = zeros(1,1,1,length(gtids)*numBB, 'single');

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
    
    % Perform Selective Search
    [boxes] = Image2HierarchicalGrouping(img, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    
    % Randomly select numBB boxes
    inds = randperm(size(boxes,1));
    boxes = boxes(inds(1:numBB), :); 
    boxes = boxes(:,[2 1 4 3]); % to match pascal [xmin, ymin, xmax, ymax] format
    
    % Iterate through boxes and crop specified regions
    box_labels = zeros(size(boxes,1), 1);
    for cr = 1:size(boxes,1)
        % Assess intersection over union with each ground truth object
        for ob = 1:length(recs(i).objects)
            gt_bbox = recs(i).objects(ob).bbox;
            if(IoU(boxes(cr,:), gt_bbox) > 0.5)
                % Leave label as zero (background) unless a match is found
                box_labels(cr) = find(ismember(VOCopts.classes, recs(i).objects(ob).class));
            end
        end
    end
    fore = find(box_labels ~= 0);
    back = find(box_labels == 0);
    numFore = min(5, length(fore));
    numBack = 5 + (5 - numFore);
    hits = [fore(1:numFore); back(1:numBack)];
    for h = 1:length(hits)
       cr = hits(h);
       cropped_im = img(boxes(cr,2):boxes(cr,4), boxes(cr,1):boxes(cr,3), :);
       cropped_im = imresize(cropped_im, [imgsize imgsize]); % eliminated im2double
       data(:,:,:,count) = cropped_im;
       labels(:,:,:,count) = box_labels(cr);
       count = count + 1;
    end
end



%% Save tensors
saveDir = '/home/aseff/marvin/rcnn/';

tensor.type = 'half';
tensor.sizeof = 2;
tensor.name = 'regions';
tensor.value = single(data);
tensor.dim = 4;
writeTensors(sprintf('%s/regions_half.tensor', saveDir), tensor);

fakelabels = zeros(1,1,1,size(data, 4));
tensor.value = single(fakelabels);
writeTensors(sprintf('%s/regions_fakelabels_half.tensor', saveDir), tensor);
toc

%% Run AlexNet on extracted regions
cd('/home/aseff/marvin');
setenv('LD_LIBRARY_PATH', '/usr/local/cuda/lib64/:/usr/local/cudnn/v4rc/lib64');
setenv('PATH', '/usr/local/cuda/bin/');
system('./marvin test rcnn/alexnet_imagenet.json models/alexnet_imagenet/alexnet_imagenet_half.marvin loss rcnn/prob_file.tensor');


