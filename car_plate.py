#!/usr/bin/env python
# coding: utf-8

# This code is based on 
# 
# https://github.com/matterport/Mask_RCNN/blob/v2.1/samples/balloon/balloon.py
# 
# "Mask R-CNN
# Train on the toy Balloon dataset and implement color splash effect.
# Copyright (c) 2018 Matterport, Inc.
# Licensed under the MIT License (see LICENSE for details)
# Written by Waleed Abdulla"
# 
# This project is a transfer learning based on Matterport's model. I used the pre-trained model as a start point to retrain my own model for new object detection ("car_plate"). Before run the code, please go to the matterport website below for installation information first.
# 
# https://github.com/matterport/Mask_RCNN/tree/v2.1
# 
# 
# 

# "Mask R-CNN (regional convolutional neural network) is a two stage framework: the first stage scans the image and generates proposals(areas likely to contain an object). And the second stage classifies the proposals and generates bounding boxes and masks."
# 
# Mask-RCNN consists of the following modules:
# 1. Backbone: 
# 
#    1.1 ResNet: ResNet CNN for feature extraction, early layers for edges/corners and later layers for person/sky/car/... Passing throught the backbone, the image is converted from 1024x1024x3 to 32x32x2048
#    
#    1.2 Feature Pyramid Network: The FPN is used as the ResNet extension, which can better represent objects at multiple scales. "Faster R-CNN with an FPN backbone extracts RoI features from different levels of the feature pyramid according to their scale. Using a ResNet-FPN backbone for feature extraction with Mask RCNNgives excellent gains in both accuracy and speed." It allows features at every level to have access to both, lower and higher level features.
# 
# 2. Region Proposal Network (RPN): The RPN is a lightweight neural network that scans the image in a sliding-window fashion and finds areas that contain objects. RPN doesn’t scan over the image directly (even though we draw the anchors on the image for illustration). Instead, the RPN scans over the backbone feature map. The RPN generates 2 outputs for each anchor: Anchor class and bounding box refinement.
# 
#     Anchor Class: one of the two classes: foreground and background. Foreground class implies a possible object in that box.
#     
#     Bounding box refinement: RPN estimate a delta to refine the anchor box to fit the object better.
#     
#     If several anchors overlap too much, we keep the one with the highest foreground score and discard the rest (referred to as Non-max Suppression). After that we have the final proposals (regions of interest) that we pass to the next stage.
#     
# 3. ROI Classifier and Bounding Box Regressor: It generate 2 outputs
#     
#     Class of the object in the ROI
#     
#     Bounding Box Refinement
#     
#     ROI pooling refers to cropping a part of a feature map and resizing it to a fixed size. This enables ROI to have various sizes as inputs for classifier.
#     
# 4. Segmentation Masks
# 
#     The mask branch is a convolutional network that takes the positive regions selected by the ROI classifier and generates masks for them. During training, we scale down the ground-truth masks to 28x28 to compute the loss, and during inferencing we scale up the predicted masks to the size of the ROI bounding box and that gives us the final masks, one per object.
# 
# 

# Object detection is a challenging problem that involves object recognition (e.g. where are they), object localization (e.g. what are their extent), and object classification (e.g. what are they).
# 
# R-CNN: Bounding boxes are proposed by the “selective search” algorithm, each of which is stretched and features are extracted via a deep convolutional neural network, such as AlexNet, before a final set of object classifications are made with linear SVMs.
# 
# Fast R-CNN: Simplified design with a single model, bounding boxes are still specified as input, but a region-of-interest pooling layer is used after the deep CNN to consolidate regions and the model predicts both class labels and regions of interest directly.
# 
# Faster R-CNN: Addition of a Region Proposal Network that interprets features extracted from the deep CNN and learns to propose regions-of-interest directly.
# 
# Mask R-CNN: Extension of Faster R-CNN that adds an output model for predicting a mask for each detected object.
# 
# "The Region-based CNN (R-CNN) approach to bounding-box object detection is to attend to a manageable number of candidate object regions and evaluate convolutional networks independently on each RoI. R-CNN was extended to allow attending to RoIs on feature maps using RoIPool, leading to fast speed and better accuracy. Faster R-CNN advanced this stream by learning the attention mechanism with a Region Proposal Network (RPN). Faster R-CNN is flexible and robust to many follow-up improvements, and is the current leading framework in several benchmarks."
# 
# 

# In[ ]:


import os
import sys
import json
import numpy as np
import skimage.draw


# In[ ]:


ROOT_DIR = "./"

# config Name, GPU use, validation/train steps per epoch, backbone architecture(resnet50 or resnet101), num of clf class,
# anchors sizes, etc. 
from mrcnn.config import Config

# utils contains tools for bounding boxes, non-max suppression, box refinement, database functions, and anchor functions, etc.
from mrcnn import utils


# In[ ]:


# Directory to save logs and model checkpoints, if not providedt
DEFAULT_LOGS_DIR = ROOT_DIR + "logs"


# In[ ]:


############################################################
#  Configurations
############################################################


class car_plateConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "car_plate"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + car_plate

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

config = car_plateConfig()
config.display()


# In[ ]:


os.getcwd()


# In[ ]:


############################################################
#  Dataset
############################################################

class car_plateDataset(utils.Dataset):

    def load_car_plate(self, dataset_dir, subset):
        """Load a subset of the car_plate dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("car_plate", 1, "car_plate")

        # Train or validation dataset?
        assert subset in ["train", "dev"]
        dataset_dir = dataset_dir + '/' + subset + '/'
        
        # Load annotations
        annotations = json.load(open(dataset_dir + "car_plate_via.json"))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = annotations[1]
        annotations = list(annotations.values())
            

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = dataset_dir + a['filename']
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "car_plate",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a car_plate dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "car_plate":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "car_plate":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


# In[ ]:


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = car_plateDataset()
    dataset_train.load_car_plate("dataset", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = car_plateDataset()
    dataset_val.load_car_plate("dataset", "dev")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


# In[ ]:


from mrcnn.model import MaskRCNN
model = MaskRCNN(mode="training", config=config,
                          model_dir="model.py")


# In[ ]:


model.load_weights("mask_rcnn_balloon.h5", by_name=True)


# In[ ]:


train(model)


# In[ ]:


class PredictionConfig(Config):
    NAME = "car_plate_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# In[ ]:


cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)


# In[ ]:


model_path = 'mask_rcnn_car_plate_0020.h5'
model.load_weights(model_path, by_name=True)


# In[ ]:


from mrcnn.model import load_image_gt
# load image, bounding boxes and masks for the image id


# In[ ]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
img = load_img('car_plate_test_2.jpg')
img = img_to_array(img)


# In[ ]:


from mrcnn.visualize import display_instances
results = model.detect([img], verbose = 0)
# get dictionary for first prediction
r = results[0]
class_names = ['BG','car_plate']
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


# In[ ]:




