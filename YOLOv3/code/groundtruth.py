# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:54:39 2019

@author: LiXiaoGang
"""
from __future__ import division

import math
import numpy as np
import parameters as para
from parse import parse_size
from parse import parse_object
from onehotcode import onehotencode


def get_groundtruth(xml):
    gt_loc = []
    gt_cls = []
    masks = []
    gt_box = []
    ind_cx = []
    ind_cy =[]
    feat_scale = []
    prior_anchors = []
    truth_array = np.zeros([4*para.MAX_NUM_GT],dtype=np.float32)
    
    size_dict = parse_size(xml)
    object_list = parse_object(xml)
    rw = 1.0*para.INPUT_SIZE/size_dict['width']
    rh = 1.0*para.INPUT_SIZE/size_dict['height']
    
    grid_cell_size = [para.INPUT_SIZE/para.FEATURE_MAPS[0][0],
                      para.INPUT_SIZE/para.FEATURE_MAPS[1][0],
                      para.INPUT_SIZE/para.FEATURE_MAPS[2][0]]
    
    for s in range(len(para.FEATURE_MAPS)):
        shape1 = (para.FEATURE_MAPS[s][0],para.FEATURE_MAPS[s][1])
        shape2 = (para.FEATURE_MAPS[s][0],para.FEATURE_MAPS[s][1],para.BOXES*(4+1))
        shape3 = (para.FEATURE_MAPS[s][0],para.FEATURE_MAPS[s][1],para.BOXES*para.NUM_CLASSESS)
        
        mask = np.zeros( shape1, dtype=np.float32)
        ind_x = np.zeros( shape1, dtype=np.float32)
        ind_y = np.zeros( shape1, dtype=np.float32)   
        bbox = np.zeros( shape2, dtype=np.float32)
        anch = np.zeros( shape2, dtype=np.float32)
        loc = np.zeros( shape2, dtype=np.float32)
        cls = np.zeros( shape3, dtype=np.float32)
        
        for i in range(para.FEATURE_MAPS[s][1]):
            for j in range(para.FEATURE_MAPS[s][0]):
                anchors = []
                for k in range(para.BOXES*s,para.BOXES*(s+1)):                    
                    pw = para.ANCHORS.Width[k]
                    ph = para.ANCHORS.Height[k]
                    anchor = [0.0,0.0,pw,ph,0.0]
                    anchors = np.hstack((anchors,anchor))
                anchors = np.array(anchors,dtype=np.float32)[None]
                anch[i,j,:] = anchors         
                
        j = 0
        for box in object_list:
            box_class = box['classes']
            xmin =  box['xmin']*rw
            ymin =  box['ymin']*rh
            xmax =  box['xmax']*rw
            ymax =  box['ymax']*rh
            
            x_center = xmin + (xmax-xmin)/2.0
            y_center = ymin + (ymax-ymin)/2.0
            
            cx = x_center/(para.INPUT_SIZE)
            cy = y_center/(para.INPUT_SIZE)
                              
            bx = 1.0*x_center/grid_cell_size[s]
            by = 1.0*y_center/grid_cell_size[s]
            bw = 1.0*(xmax-xmin)/(para.INPUT_SIZE)
            bh = 1.0*(ymax-ymin)/(para.INPUT_SIZE)
            
            x_cell = int(bx)
            y_cell = int(by)
            
            obj = 1.0    # objectness score
            bboxes = []
            classes = []
            anchors = []
            offsets = []
            class_onehotcode = np.squeeze(onehotencode([box_class+'_*']))
            for i in range(para.BOXES*s,para.BOXES*(s+1)):
                pw = para.ANCHORS.Width[i]
                ph = para.ANCHORS.Height[i]
                
                # need to predict
                sigmoid_tx = max((bx - x_cell),1e-10)
                sigmoid_ty = max((by - y_cell),1e-10)
                tw = math.log(bw/pw)
                th = math.log(bh/ph)
                
                anchor = [cx,cy,pw,ph,obj]
                box = [cx,cy,bw,bh,obj]
                offset = [sigmoid_tx,sigmoid_ty,tw,th,obj]
                
                bboxes = np.hstack((bboxes,box))
                offsets = np.hstack((offsets,offset))
                anchors = np.hstack((anchors,anchor))
                classes = np.hstack((classes,class_onehotcode))
                                
            bboxes = np.array(bboxes,dtype=np.float32)[None]
            offsets = np.array(offsets,dtype=np.float32)[None]
            classes = np.array(classes,dtype=np.float32)[None]
            anchors = np.array(anchors,dtype=np.float32)[None]
            
            bbox[y_cell,x_cell,:] = bboxes
            loc[y_cell,x_cell,:] = offsets
            cls[y_cell,x_cell,:] = classes
            anch[y_cell,x_cell,:] = anchors   
            mask[y_cell,x_cell] = 1.0
            
            if (j < para.MAX_NUM_GT) and (s == 0):
                truth_array[4*j:4*(j+1)] = np.divide([xmin,ymin,xmax,ymax],para.INPUT_SIZE,dtype=np.float32)
                j = j + 1
        
        bbox = np.reshape(bbox,(para.FEATURE_MAPS[s][0]*para.FEATURE_MAPS[s][1],para.BOXES*(4+1)))
        loc = np.reshape(loc,(para.FEATURE_MAPS[s][0]*para.FEATURE_MAPS[s][1],para.BOXES*(4+1)))
        cls = np.reshape(cls,(para.FEATURE_MAPS[s][0]*para.FEATURE_MAPS[s][1],para.BOXES*para.NUM_CLASSESS))
        anch = np.reshape(anch,(para.FEATURE_MAPS[s][0]*para.FEATURE_MAPS[s][1],para.BOXES*(4+1)))
        mask = np.reshape(mask,(para.FEATURE_MAPS[s][0]*para.FEATURE_MAPS[s][1],1))
        
        
        for idy in range(para.FEATURE_MAPS[s][1]):
            for idx in range(para.FEATURE_MAPS[s][0]):
                ind_x[idy,idx] = idx
                     
        for idx in range(para.FEATURE_MAPS[s][0]):
            for idy in range(para.FEATURE_MAPS[s][1]):
                ind_y[idy,idx] = idy
                     
        ind_x = np.reshape(ind_x,(para.FEATURE_MAPS[s][0]*para.FEATURE_MAPS[s][1],1))
        ind_y = np.reshape(ind_y,(para.FEATURE_MAPS[s][0]*para.FEATURE_MAPS[s][1],1))
        
        scale = np.ones(shape1, dtype=np.float32)*para.FEATURE_MAPS[s][0]
        scale = np.reshape(scale,(para.FEATURE_MAPS[s][0]*para.FEATURE_MAPS[s][1],1))

        gt_box.append(bbox)
        gt_loc.append(loc)
        gt_cls.append(cls)
        masks.append(mask)
        ind_cx.append(ind_x)
        ind_cy.append(ind_y)        
        prior_anchors.append(anch)
        feat_scale.append(scale)
    
    gt_box = np.vstack(gt_box)
    gt_loc = np.vstack(gt_loc)
    gt_cls = np.vstack(gt_cls)
    masks = np.vstack(masks)
    ind_cx = np.vstack(ind_cx)
    ind_cy = np.vstack(ind_cy) 
    prior_anchors = np.vstack(prior_anchors)
    feat_scale = np.vstack(feat_scale)
    
    gt_info = {'loc':gt_loc,'cls':gt_cls,'mask':masks,'anchors':prior_anchors,'box':gt_box,'ind_cx':ind_cx,'ind_cy':ind_cy,'feat_scale':feat_scale,'gtruth':truth_array}
    return gt_info