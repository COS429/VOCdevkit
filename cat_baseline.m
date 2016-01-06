function cat_baseline

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% loading ground truth validation
load('local/VOC2012/val_anno.mat');

% initialize VOC options
VOCinit;

% train and test detector for each class
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};
    detector=train(VOCopts,cls);                            % train detector
    test(VOCopts,cls,detector, recs);                             % test detector
    [recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);  % compute and display PR
    
    if i<VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        drawnow;
        pause;
    end
end
end

% train detector
function detector = train(VOCopts,cls)
detector = 0;
end

% run detector on test images
function out = test(VOCopts,cls,detector, recs)

% load test set ('val' for development kit)
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');

% create results file
fid=fopen(sprintf(VOCopts.detrespath,'comp3',cls),'w');

% apply detector to each image
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end
    
    % find objects of class
    clsinds=strmatch(cls,{recs(i).objects(:).class},'exact');
    notclsinds = find(~ismember(1:length(recs(i).objects), clsinds));
    
    % assign ground truth class to image
    if ~isempty(clsinds)
        
         BB = cat(1,recs(i).objects(clsinds).bbox)';
         c = randn(1, size(BB,2)) + .5 + eps;
    else
        BB = cat(1, recs(i).objects(notclsinds).bbox)';
        c = randn(1, size(BB,2)) + eps;
    end
    
    % write to results file
    for j=1:length(c)
        fprintf(fid,'%s %f %f %f %f %f\n',ids{i},c(j),BB(:,j));
    end
end

% close results file
fclose(fid);
end

% % trivial detector: confidence is computed as in example_classifier, and
% % bounding boxes of nearest positive training image are output
% function [c,BB] = detect(VOCopts,detector,I)
% 
% % compute confidence
% c = rand + eps;  %always
% 
% halfsize = 100;
% xmin = round(size(I,2)/2) - halfsize;
% ymin = round(size(I,1)/1) - halfsize;
% xmax = round(size(I,2)/2) + halfsize;
% ymax = round(size(I,1)/1) + halfsize;
% 
% % copy bounding boxes from nearest positive image
% BB=[xmin, ymin, xmax, ymax]';
% end