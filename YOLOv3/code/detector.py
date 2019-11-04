# -*- coding: utf-8 -*-

import parameters as para
import tensorflow as tf


def conv2d(inputs,filters,kernel_size,stride,mode):
    net = tf.layers.conv2d(inputs=inputs,filters=filters,kernel_size=kernel_size,strides=(stride,stride),padding='same',activation=None,use_bias=False)
    net = tf.layers.batch_normalization(net,training=mode)
    net = tf.nn.leaky_relu(net,alpha=0.1)
    return net


def conv2d_block(inputs,filters,num_blocks,mode):
    k_size = [(1,1),(3,3),(1,1),(3,3),(1,1)]
    for i in range(num_blocks):
        inputs = conv2d(inputs,filters,k_size[i],1,mode)
    return inputs


def residual_block(inputs,filters,num_blocks,mode):
    for i in range(num_blocks):
        net = conv2d(inputs,filters,(1,1),1,mode)
        net = conv2d(net,2*filters,(3,3),1,mode)
        inputs = net + inputs
    return inputs


def YOLODetector(inputs,mode):
    '''
    Implementation of Darknet-53.
    Architecture: https://arxiv.org/abs/1804.02767
    '''
    with tf.variable_scope('YOLOv3Detector'):
        with tf.variable_scope('FeatureExtractor/Darknet53'):
            net = conv2d(inputs,32,(3,3),1,mode)
            net = conv2d(net,64,(3,3),2,mode)
            net = residual_block(net,32,1,mode)
            net = conv2d(net,128,(3,3),2,mode)
            net = residual_block(net,64,2,mode)
            net = conv2d(net,256,(3,3),2,mode)
            net1 = residual_block(net,128,8,mode)   # output size:52 x 52
            net = conv2d(net1,512,(3,3),2,mode)
            net2 = residual_block(net,256,8,mode)   # output size:26 x 26
            net = conv2d(net2,1024,(3,3),2,mode)
            net3 = residual_block(net,512,8,mode)   # output size:13 x 13
    
        with tf.variable_scope('Detectors'):
            with tf.variable_scope('Detector3'):
                net_d3 = conv2d_block(net3,1024,5,mode)
                net = conv2d(net_d3,para.OUYPUT_CHANNELS,(3,3),1,mode)
                
                loc3 = tf.layers.conv2d(net,para.BOXES*(4+1),(1,1),(1,1),padding='same',activation=None,use_bias=True)
                loc3 = tf.reshape(loc3,shape=[-1,para.FEATURE_MAPS[2][0]*para.FEATURE_MAPS[2][1],para.BOXES*(4+1)])
                
                cls3 = tf.layers.conv2d(net,para.BOXES*para.NUM_CLASSESS,(1,1),(1,1),padding='same',activation=None,use_bias=True)
                cls3 = tf.reshape(cls3,shape=[-1,para.FEATURE_MAPS[2][0]*para.FEATURE_MAPS[2][1],para.BOXES*para.NUM_CLASSESS])
                
            with tf.variable_scope('Detector2'):
                net = conv2d(net_d3,256,(1,1),1,mode)
                _,s1,s2,_ = net2.get_shape().as_list()
                net = tf.image.resize_bilinear(net,(s1,s2))
                net = tf.concat([net,net2],axis=3)
                net_d2 = conv2d_block(net,256,5,mode)
                net = conv2d(net_d2,para.OUYPUT_CHANNELS,(3,3),1,mode)
                
                loc2 = tf.layers.conv2d(net,para.BOXES*(4+1),(1,1),(1,1),padding='same',activation=None,use_bias=True)
                loc2 = tf.reshape(loc2,shape=[-1,para.FEATURE_MAPS[1][0]*para.FEATURE_MAPS[1][1],para.BOXES*(4+1)])
                
                cls2 = tf.layers.conv2d(net,para.BOXES*para.NUM_CLASSESS,(1,1),(1,1),padding='same',activation=None,use_bias=True)
                cls2 = tf.reshape(cls2,shape=[-1,para.FEATURE_MAPS[1][0]*para.FEATURE_MAPS[1][1],para.BOXES*para.NUM_CLASSESS])
                
            with tf.variable_scope('Detector1'):
                net = conv2d(net_d2,128,(1,1),1,mode)
                _,s1,s2,_ = net1.get_shape().as_list()
                net = tf.image.resize_bilinear(net,(s1,s2))
                net = tf.concat([net,net1],axis=3)
                net_d1 = conv2d_block(net,128,5,mode)
                net = conv2d(net_d1,para.OUYPUT_CHANNELS,(3,3),1,mode)
                
                loc1 = tf.layers.conv2d(net,para.BOXES*(4+1),(1,1),(1,1),padding='same',activation=None,use_bias=True)
                loc1 = tf.reshape(loc1,shape=[-1,para.FEATURE_MAPS[0][0]*para.FEATURE_MAPS[0][1],para.BOXES*(4+1)])
                
                cls1 = tf.layers.conv2d(net,para.BOXES*para.NUM_CLASSESS,(1,1),(1,1),padding='same',activation=None,use_bias=True)
                cls1 = tf.reshape(cls1,shape=[-1,para.FEATURE_MAPS[0][0]*para.FEATURE_MAPS[0][1],para.BOXES*para.NUM_CLASSESS])
            
            loc = tf.concat([loc1,loc2,loc3],axis=1,name='Localization')
            cls = tf.concat([cls1,cls2,cls3],axis=1,name='Classification')
    return loc,cls
