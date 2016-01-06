function [allBoxes, allLabels, recs_test] = roi

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

set = VOCopts.testset;
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

allBoxes = [];
allLabels = [];

numBB = 10;
% data = zeros(600, 600, 3, length(gtids)*numBB, 'single');
% labels = cell(length(gtids), 1);
% bboxes = labels;
tic;
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('load actual: %d/%d\n',i,length(gtids));
    end
    
    %     % load image
    %     img=imread(sprintf(VOCopts.imgpath,gtids{i}));
    %     img600 = imresize(img,[600,600]);
    %     xScale = size(img,2)*1.0/600; % x is columns
    %     yScale = size(img,1)*1.0/600;
    %     data(:,:,:,i) = single(img600);
    
    % find all objects in each image
    for ob = 1:length(recs(i).objects)
        bbox = recs(i).objects(ob).bbox;
        X = recs(i).size.width; Y = recs(i).size.height;
        bbox = [bbox(1)/X, bbox(2)/Y, bbox(3)/X, bbox(4)/Y];
        allBoxes(end+1,:) = bbox;
        allLabels(end+1,1) = find(ismember(VOCopts.classes, recs(i).objects(ob).class));
        %         bbox = [bbox(1)/xScale, bbox(2)/yScale, bbox(3)/xScale, bbox(4)/yScale];
        %         bboxes{i}(end+1,:) = bbox;
        %         labels{i}(end+1) = find(ismember(VOCopts.classes, recs(i).objects(ob).class)) - 1;
    end
end

recs_test = recs;

end