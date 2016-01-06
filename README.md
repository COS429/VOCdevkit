# VOCdevkit

bb_prepare.m loads all Pascal images, runs Selective Search on each image to obtain candidate bounding boxes, and assigns ground truth labels to each bounding box based on IoU (with a ground truth bounding box). All of these are then stored in tensor format for subsequent use by Marvin.
