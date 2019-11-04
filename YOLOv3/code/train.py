# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:46:14 2019
https://blog.csdn.net/xiaohu2022/article/details/80666655
https://blog.csdn.net/qq_37541097/article/details/81214953

@author: LiXiaoGang        
"""
from __future__ import division

import os
import sys
import shutil
import tensorflow as tf
import parameters as para
from loss import YOLOLoss
from detector import YOLODetector
from readbatch import mini_batch
from postprocessing import nms,box_decode,save_instance


def net_placeholder(input_size,channel,num_cells,boxes,num_classes,max_num_gt,batch_size=None):     
    images = tf.placeholder(dtype=tf.float32,shape=[batch_size,input_size,input_size,channel],name='input')
    gt_cls = tf.placeholder(dtype=tf.float32,shape=[batch_size,num_cells,boxes*num_classes])
    gt_loc = tf.placeholder(dtype=tf.float32,shape=[batch_size,num_cells,boxes*(4+1)])
    gt_box = tf.placeholder(dtype=tf.float32,shape=[batch_size,num_cells,boxes*(4+1)])
    anchors= tf.placeholder(dtype=tf.float32,shape=[batch_size,num_cells,boxes*(4+1)])
    
    masks = tf.placeholder(dtype=tf.float32,shape=[batch_size,num_cells,1])
    ind_cx= tf.placeholder(dtype=tf.float32,shape=[batch_size,num_cells,1])
    ind_cy= tf.placeholder(dtype=tf.float32,shape=[batch_size,num_cells,1])
    gtruth = tf.placeholder(dtype=tf.float32,shape=[batch_size,4*max_num_gt])
    feat_scale = tf.placeholder(dtype=tf.float32,shape=[batch_size,num_cells,1])
    isTraining = tf.placeholder(tf.bool,name='batchnorm')
    return images,gt_loc,gt_cls,masks,anchors,gt_box,ind_cx,ind_cy,feat_scale,gtruth,isTraining


def training_net():
    images,gt_loc,gt_cls,masks,anchors,gt_box,ind_cx,ind_cy,feat_scale,gtruth,isTraining = net_placeholder(para.INPUT_SIZE,para.CHANNEL,para.NUM_CELLS,para.BOXES,
                                                                                                           para.NUM_CLASSESS,para.MAX_NUM_GT,batch_size=None,)
    pred_loc,pred_cls = YOLODetector(images,isTraining)
    loss = YOLOLoss(pred_loc,pred_cls,gt_loc,gt_cls,masks,anchors,gt_box,gtruth,ind_cx,ind_cy,feat_scale)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):        
        train_step = tf.train.AdamOptimizer(para.LEARNING_RATE).minimize(loss)
     
    Saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=5)
    with tf.Session() as sess:
        
#        writer = tf.summary.FileWriter(os.path.join(para.PATH,'model'), sess.graph)
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        # restore model 
        if para.RESTORE_MODEL:
            if not os.path.exists(para.CHECKPOINT_MODEL_SAVE_PATH):
                print('Model does not exist！')
                sys.exit()
            ckpt = tf.train.get_checkpoint_state(para.CHECKPOINT_MODEL_SAVE_PATH)
            model = ckpt.model_checkpoint_path.split('\\')[-1]
            Saver.restore(sess,os.path.join(para.CHECKPOINT_MODEL_SAVE_PATH,'.\\'+ model))
            print('Successfully restore model:',model)
      
        for i in range(para.TRAIN_STEPS):
            batch = mini_batch(i,para.BATCH_SIZE,'train')
            feed_dict = {images:batch['image'],gt_loc:batch['gt_loc'],gt_cls:batch['gt_cls'],masks:batch['mask'],anchors:batch['anchors'],
                         gt_box:batch['gt_box'],ind_cx:batch['ind_cx'],ind_cy:batch['ind_cy'],feat_scale:batch['feat_scale'],gtruth:batch['gtruth'],isTraining:True}
            _,loss_ = sess.run([train_step,loss],feed_dict=feed_dict)
            print('===>Step %d: loss = %g ' % (i,loss_))
            
            # evaluate and save checkpoint
            if i % 100 == 0:
                write_instance_dir = os.path.join(para.PATH,'pic')
                if not os.path.exists(write_instance_dir):os.mkdir(write_instance_dir)
                j = 0
                while True:
                    
                    batch = mini_batch(j,1,'val')
                    feed_dict = {images:batch['image'],isTraining:False}
                    location,confidence = sess.run([pred_loc,pred_cls],feed_dict=feed_dict)
                    predictions = {'location':location,'confidence':confidence}
                    
                    pred_output = box_decode(predictions,batch['ind_cx'],batch['ind_cy'],batch['feat_scale'],batch['anchors'],para.CONFIDENCE_THRESHOLD,batch['image_name'])
                    pred_output = nms(pred_output,para.NMS_THRESHOLD)
                    
                    if j < min(10,batch['image_num']):save_instance(pred_output)
                    if j == batch['image_num']-1:break
                    j += 1
                
                if os.path.exists(para.CHECKPOINT_MODEL_SAVE_PATH) and (i==0):
                    shutil.rmtree(para.CHECKPOINT_MODEL_SAVE_PATH)
                Saver.save(sess,os.path.join(para.CHECKPOINT_MODEL_SAVE_PATH,para.MODEL_NAME))             
            

def main():
    training_net()
     
if __name__ == '__main__':
    main()