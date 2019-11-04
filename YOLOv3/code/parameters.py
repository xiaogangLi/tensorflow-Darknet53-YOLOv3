# -*- coding: utf-8 -*-

from __future__ import division

import os
import pandas as pd


PATH = os.path.dirname(os.getcwd())
LABELS = pd.read_csv(os.path.join(PATH,'label','label.txt'))

# Training para
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
TRAIN_STEPS = 500000
PIC_TYPE = 'jpg'             # the picture format of training images.
RESTORE_MODEL = False
MAX_NUM_GT = 10              # Suppose that each image contains up to 10 objects

# K-Means para
NUM_CLUSTER = 9    
MAX_ITERS = 20

# YOLOv3 para
CHANNEL = 3
INPUT_SIZE = 416
OBJECT_SCALE = 5
NOOBJECT_SCALE = 0.1
CLASS_SCALE = 1
COORD_SCALE =1
MAX_IOU = 0.5
FEATURE_MAPS = [(52,52),(26,26),(13,13)]
CONFIDENCE_THRESHOLD = 0.5                 # background when objectness score < 0.5
NMS_THRESHOLD = 0.3
NUM_ANCHORS = NUM_CLUSTER                  # YOLOv3 will predict NUM_ANCHORS//3 bounding boxes in each grid cell.
NUM_CLASSESS = len(LABELS.Class_name)
BOXES = NUM_ANCHORS//3                     # predict 3 boxes at each scale
OUYPUT_CHANNELS = BOXES*(4+1+NUM_CLASSESS)
NUM_CELLS = FEATURE_MAPS[0][0]**2+FEATURE_MAPS[1][0]**2+FEATURE_MAPS[2][0]**2
                                 
ANCHORS = pd.read_csv(os.path.join(PATH,'anchor','anchor.txt'))
MODEL_NAME = 'model.ckpt'
CHECKPOINT_MODEL_SAVE_PATH = os.path.join(PATH,'model','checkpoint')
assert NUM_ANCHORS % BOXES == 0
