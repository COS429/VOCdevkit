# VOCdevkit

bb_prepare.m prepares PASCAL data for subsequent use by Marvin via the following steps:

1. Loads all Pascal images.
   - These are saved under /data/cos429/VOCdevkit/ on visiongpu.
   
2. Runs Selective Search on each image.
   - This returns a set of candidate bounding boxes for each image (as in Fast R-CNN).
   - These bounding boxes will be used by Marvin's ROIPooling layer.

3. Assigns ground truth labels to each bounding box.
   - If a candidate box has an IoU with a ground truth box > 0.5, it is assigned the label of the ground truth object.
  
4. Rescales Pascal images to 227 x 227.
   - The longer dimension of each image is rescaled to 227, and the other dimension is adjusted to maintain the aspect ratio.
   - The images are then zero-padded to be 227 x 227.
   - The bounding box coordinates are also scaled accordingly.
   
5. Saves all information above in Marvin tensor format.
