# -*- coding: utf-8 -*-

from __future__ import division

import os
import math
import datetime
import cv2 as cv
import numpy as np
import parameters as para
from onehotcode import onehotdecode


def sigmoid(x):
    return 1/(1+np.exp(-x))


def correct_location(x):
    x = min(max(x,0),1)
    return x


def box_decode(predictions,ind_cx,ind_cy,feat_scale,anchors,threshold,imgname):
    pred_loc = np.squeeze(predictions['location'])
    pred_cls = np.squeeze(predictions['confidence'])
    ind_cx = np.squeeze(ind_cx)
    ind_cy = np.squeeze(ind_cy)
    feat_scale = np.squeeze(feat_scale)
    anchors = np.squeeze(anchors)
    
    boxes = []
    for i in range(para.NUM_CELLS):
        anchor = anchors[i,:]
        pre_class = pred_cls[i,:]
        pred_offset = pred_loc[i,:]
        cx = ind_cx[i]
        cy = ind_cy[i]
        scale = feat_scale[i]
        
        for j in range(para.BOXES):
            anchor_w = anchor[5*j+2]
            anchor_h = anchor[5*j+3]
                    
            pred_obj_sig = pred_offset[5*j+4]
            pred_box_offset = pred_offset[5*j:5*j+4]
            pred_box_prob = sigmoid(pre_class[4*j:4*j+4])
            
            bx = (pred_box_offset[0] + cx)/scale
            by = (pred_box_offset[1] + cy)/scale
            bw = math.exp(pred_box_offset[2])*anchor_w
            bh = math.exp(pred_box_offset[3])*anchor_h
                         
            obj_score = sigmoid(pred_obj_sig)
            if obj_score > threshold:                
                max_prob = max(pred_box_prob)
                pred_classname = onehotdecode(pred_box_prob)
                pred_xmin = correct_location(bx - bw*0.5)
                pred_ymin = correct_location(by - bh*0.5)
                pred_xmax = correct_location(bx + bw*0.5)
                pred_ymax = correct_location(by + bh*0.5)
                
                pred_box = {'box':[pred_xmin,pred_ymin,pred_xmax,pred_ymax,max_prob],'className':pred_classname}
                boxes.append(pred_box)
    result = {'imageName':imgname,'boxes':boxes}
    return result


def calculateIoU(xmin0,ymin0,xmax0,ymax0,xmin1,ymin1,xmax1,ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    intersection = w*h
    union = (xmax0-xmin0)*(ymax0-ymin0)+(xmax1-xmin1)*(ymax1-ymin1)-intersection
    
    if union<=0.0:
        iou = 0.0
    else:
        iou = 1.0*intersection / union
    return iou


def nms(result,threshold):
    class_list =[]
    final_pred_boxes = []
    boxes = result['boxes']
    
    for b in range(len(boxes)):
        class_list.append(boxes[b]['className'])
    class_list = np.unique(class_list)
    
    for name in class_list:
        box_coord = []
        for b in range(len(boxes)):
            if name == boxes[b]['className']:
                box_coord.append(boxes[b]['box'])       
        box_coord = np.array(box_coord)
        
        while box_coord.shape[0] > 0:                
            idx = np.argmax(box_coord[:,-1])
            keep_box = box_coord[idx,:]
            pred_box = {'box':keep_box,'className':name}
            final_pred_boxes.append(pred_box)
            
            box_coord = np.delete(box_coord,[idx],axis=0)
            if box_coord.shape[0] == 0:break
            
            suppre = []
            xmin0 = keep_box[0]
            ymin0 = keep_box[1]
            xmax0 = keep_box[2]
            ymax0 = keep_box[3]
            
            for b in range(box_coord.shape[0]):
                xmin1 = box_coord[b,:][0]
                ymin1 = box_coord[b,:][1]
                xmax1 = box_coord[b,:][2]
                ymax1 = box_coord[b,:][3]
                
                iou = calculateIoU(xmin0,ymin0,xmax0,ymax0,
                                   xmin1,ymin1,xmax1,ymax1)
                if iou > threshold:
                    suppre.append(b)
            box_coord = np.delete(box_coord,suppre,axis=0)
    detections = {'imageName':result['imageName'],'boxes':final_pred_boxes}
    return detections


def save_instance(detections):
    image_name = detections['imageName'][0]+'.'+para.PIC_TYPE
    read_dir = os.path.join(para.PATH,'data','annotation','images',image_name)
    write_dir = os.path.join(para.PATH,'pic')
    
    im = cv.imread(read_dir).astype(np.float32)
    im_h = im.shape[0]
    im_w = im.shape[1]
    
    im = cv.resize(im,(para.INPUT_SIZE,para.INPUT_SIZE)).astype(np.float32)
    for b in range(len(detections['boxes'])):
        box = detections['boxes'][b]['box']
        name = detections['boxes'][b]['className']
        
        xmin = int(box[0]*para.INPUT_SIZE)
        ymin = int(box[1]*para.INPUT_SIZE)
        xmax = int(box[2]*para.INPUT_SIZE)
        ymax = int(box[3]*para.INPUT_SIZE)
        prob = min(round(box[4]*100),100.0)
        txt = name +':'+ str(prob) + '%'
        
        font = cv.FONT_HERSHEY_PLAIN
        im = cv.rectangle(im,(xmin,ymin),(xmax,ymax),(255, 0, 0),1)
        im = cv.putText(im,txt,(xmin,ymin),font,1,(255,0,0),1)
    
    im = cv.resize(im,(im_w,im_h)).astype(np.float32)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-')
    dst = os.path.join(write_dir,current_time+image_name)
    cv.imwrite(dst,im)
