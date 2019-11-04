# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:47:50 2019

@author: LiXiaoGang
"""

import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
import parameters as para
from groundtruth import get_groundtruth


train_image_name = pd.read_csv(os.path.join(para.PATH,'data','train','train.txt'),header=None,names=['Name'])
val_image_name = pd.read_csv(os.path.join(para.PATH,'data','val','val.txt'),header=None,names=['Name'])

def mini_batch(i,batch_size,flag):
    if flag == 'train':
        data_name = train_image_name
        
    elif flag == 'val':
        data_name = val_image_name
        
    else:
        print('The argument "%s"  does not exist!' % (flag))
        sys.exit(0)
        
    start = (i*batch_size) % len(data_name['Name'])
    end = min(start+batch_size,len(data_name['Name']))
    
    if (end - start) < batch_size:
        start = len(data_name['Name']) - batch_size
        end = len(data_name['Name'])
        
    image = np.zeros([batch_size,para.INPUT_SIZE,para.INPUT_SIZE,para.CHANNEL],dtype=np.float32)
    gt_loc = np.zeros([batch_size,para.NUM_CELLS,para.BOXES*(4+1)],dtype=np.float32)
    gt_cls = np.zeros([batch_size,para.NUM_CELLS,para.BOXES*para.NUM_CLASSESS],dtype=np.float32)
    masks = np.zeros([batch_size,para.NUM_CELLS,1],dtype=np.float32)
    anchors = np.zeros([batch_size,para.NUM_CELLS,para.BOXES*(4+1)],dtype=np.float32)
    gt_box = np.zeros([batch_size,para.NUM_CELLS,para.BOXES*(4+1)],dtype=np.float32)
    ind_cx = np.zeros([batch_size,para.NUM_CELLS,1],dtype=np.float32)
    ind_cy = np.zeros([batch_size,para.NUM_CELLS,1],dtype=np.float32)
    feat_scale = np.zeros([batch_size,para.NUM_CELLS,1],dtype=np.float32)
    gtruth = np.zeros([batch_size,4*para.MAX_NUM_GT],dtype=np.float32)
        
    batch_name = np.array(data_name['Name'][start:end])
    for j in range(len(batch_name)):
        image_name = os.path.join(para.PATH,'data','annotation','images',batch_name[j]+'.'+para.PIC_TYPE)
        im = cv.imread(image_name).astype(np.float32)
        image[j,:,:,:] = cv.resize(im,(para.INPUT_SIZE,para.INPUT_SIZE)).astype(np.float32)
        
        xml_name = os.path.join(para.PATH,'data','annotation','xml',batch_name[j]+'.xml')
        gt = get_groundtruth(xml_name)
        
        gt_loc[j,:,:] = gt['loc']
        gt_cls[j,:,:] = gt['cls']
        masks[j,:] = gt['mask']
        anchors[j,:,:] = gt['anchors']
        gt_box[j,:,:] = gt['box']
        ind_cx[j,:] = gt['ind_cx']
        ind_cy[j,:] = gt['ind_cy']
        feat_scale[j,:] = gt['feat_scale']
        gtruth[j,:] = gt['gtruth']
        
    minibatch = {'image':image,'gt_loc':gt_loc,'gt_cls':gt_cls,'mask':masks,'anchors':anchors,
                 'gt_box':gt_box,'ind_cx':ind_cx,'ind_cy':ind_cy,'feat_scale':feat_scale,
                 'gtruth':gtruth,'image_name':batch_name,'image_num':data_name.shape[0]}      
    return minibatch