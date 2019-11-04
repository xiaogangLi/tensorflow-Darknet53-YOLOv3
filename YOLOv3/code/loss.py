# -*- coding: utf-8 -*-

import tensorflow as tf
import parameters as para


def calculate_iou(xmin0,ymin0,xmax0,ymax0,xmin1,ymin1,xmax1,ymax1):
    w = tf.maximum(0.0, tf.minimum(xmax0, xmax1) - tf.maximum(xmin0, xmin1))
    h = tf.maximum(0.0, tf.minimum(ymax0, ymax1) - tf.maximum(ymin0, ymin1))
    intersection = w*h
    union = (xmax0-xmin0)*(ymax0-ymin0)+(xmax1-xmin1)*(ymax1-ymin1)-intersection       
    iou = tf.reduce_max([0.0,intersection/union])
    return iou

 
def YOLOLoss(pred_loc,pred_cls,gt_loc,gt_cls,gt_mask,anchors,gt_box,gtruth,ind_cx,ind_cy,feat_scale,eps = 1e-10):
    with tf.variable_scope('YOLOLoss'):
        losses = []
        for i in range(para.BATCH_SIZE):
            truth = gtruth[i,:]
            
            def cond(j,loss_add):
                boolean = tf.less(j,para.NUM_CELLS)
                return boolean
            
            def body(j,loss_add):
                box = gt_box[i,j,:]
                offset = gt_loc[i,j,:]
                anchor = anchors[i,j,:]
                gt_class = gt_cls[i,j,:]
                
                cx = ind_cx[i,j,0]
                cy = ind_cy[i,j,0]
                flag = gt_mask[i,j,0]
                scale = feat_scale[i,j,0]
                pre_class = pred_cls[i,j,:]
                pred_offset = pred_loc[i,j,:]
                
                iou_list = []
                loc_loss_list = []
                obj_loss_list = []
                cls_loss_list = []
                noobj_loss_list =[]
            
                for k in range(para.BOXES):
                    gt_w = box[5*k+2]
                    gt_h = box[5*k+3]
                    anchor_w = anchor[5*k+2]
                    anchor_h = anchor[5*k+3]
                                        
                    gt_box_offset = offset[5*k:5*k+4]
                    gt_box_class = gt_class[4*k:4*k+4]
                    pred_box_offset = pred_offset[5*k:5*k+4]
                    pred_obj_sig = tf.nn.sigmoid(pred_offset[5*k+4])
                    pred_cls_sig = tf.nn.sigmoid(pre_class[4*k:4*k+4])                    
                    
                    # =========================================================
                    # when the center of the object falls into the grid cell
                    iou = calculate_iou(0.0,0.0,anchor_w,anchor_h,0.0,0.0,gt_w,gt_h)
                    iou_list.append(iou)
                    
                    obj_loss = -para.OBJECT_SCALE*tf.log(tf.clip_by_value(pred_obj_sig,eps,1-eps))   # label as 1 for binary cross-entropy loss
                    obj_loss_list.append(obj_loss)
                    
                    loc_loss = para.COORD_SCALE*(2-gt_w*gt_h)*tf.reduce_sum(tf.square(tf.subtract(gt_box_offset,pred_box_offset)))
                    loc_loss_list.append(loc_loss)
                    
                    cls_loss = -para.CLASS_SCALE*tf.reduce_sum(gt_box_class*tf.log(tf.clip_by_value(pred_cls_sig,eps,1.0-eps)) + 
                                                               (1.0-gt_box_class)*tf.log(tf.clip_by_value(1.0-pred_cls_sig,eps,1.0-eps)))
                    cls_loss_list.append(cls_loss)
                    # =========================================================                    
                    
                    # =========================================================
                    # when the center of the object does not fall into the grid cell
                    bx = (pred_box_offset[0] + cx)/scale
                    by = (pred_box_offset[1] + cy)/scale
                    bw = tf.exp(pred_box_offset[2])*anchor_w
                    bh = tf.exp(pred_box_offset[3])*anchor_h
                    
                    pred_xmin = bx - bw*0.5
                    pred_ymin = by - bh*0.5
                    pred_xmax = bx + bw*0.5
                    pred_ymax = by + bh*0.5
                    
                    iou_set = []
                    for g in range(para.MAX_NUM_GT):
                        gt = truth[4*g:4*(g+1)]
                        g_xmin,g_ymin,g_xmax,g_ymax = gt[0],gt[1],gt[2],gt[3]
                        iou = calculate_iou(g_xmin,g_ymin,g_xmax,g_ymax,pred_xmin,pred_ymin,pred_xmax,pred_ymax)
                        iou_set.append(iou)
                
                    max_iou = tf.reduce_max(iou_set)                    
                    noobj_loss = -para.NOOBJECT_SCALE*tf.cast(tf.less(max_iou,para.MAX_IOU),tf.float32)*tf.log(tf.clip_by_value(1.0-pred_obj_sig,eps,1-eps))     # label as 0 for binary cross-entropy loss
                    noobj_loss_list.append(noobj_loss)
                    # =========================================================
               
                idx = tf.arg_max(iou_list,dimension=0)
                loss1 = flag*(tf.gather(loc_loss_list,idx)+tf.gather(cls_loss_list,idx)+tf.gather(obj_loss_list,idx))
                loss2 = (1.0 - flag)*tf.reduce_sum(noobj_loss_list)       # binary cross-entropy loss
                loss_add = loss_add + loss1 + loss2
                return [j+1,loss_add]
            
            j,loss_add = 0,0.0
            [j,loss_add] = tf.while_loop(cond,body,loop_vars=[j,loss_add])
            losses.append(loss_add)
        loss = tf.reduce_mean(losses,name='Loss')
    return loss

