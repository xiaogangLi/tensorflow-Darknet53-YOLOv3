# -*- coding: utf-8 -*-

from __future__ import division

import os
import numpy as np
import pandas as pd
import parameters as para
import xml.etree.ElementTree as ET
from parse import parse_object


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


def init_centroids(xmlPath,xmlName):
    names = pd.read_csv(xmlName,header=None,names=['Name'])
    fileFormat = '.xml'                                                                                             
    boxes = []
    box_area =[]
    
    for name in names.Name:
        xml = os.path.join(xmlPath,name+fileFormat)
        Tree = ET.parse(xml) 
        root = Tree.getroot()
        image_size = root.findall('size')
        image_width = int(image_size[0].find('width').text)
        image_height = int(image_size[0].find('height').text)
        object_list = parse_object(xml)
        
        for box in object_list:
            xmin =  box['xmin']
            ymin =  box['ymin']
            xmax =  box['xmax']
            ymax =  box['ymax']
             
            w = int((xmax-xmin)*(para.INPUT_SIZE/image_width))
            h = int((ymax-ymin)*(para.INPUT_SIZE/image_height))
            boxes.append((w,h))
            box_area.append(w*h)
            
    assert para.NUM_CLUSTER < len(boxes)        
    sort_area = sorted(box_area)
    step = int(len(box_area)/(para.NUM_CLUSTER+1))
    
    centroids = []
    for i in range(para.NUM_CLUSTER):
        area = sort_area[step*(i+1)]
        idx = box_area.index(area)
        centroids.append(boxes[idx])
    return {'centroids':centroids,'boxes':boxes}


def anchor_cluster(dic):
    clusters = []
    for i in range(para.NUM_CLUSTER):
        clusters.append([])
    
    centroids = dic['centroids']
    boxes = dic['boxes']
    max_iou = 0
    
    for i in range(para.MAX_ITERS):
        avg_iou = 0
        for box in boxes:
            dist_list = []
            for centroid in centroids:
                # dist(box;centroid)= 1-IOU(box;centroid)
                dist = 1 - calculateIoU(0,0,box[0],box[1],0,0,centroid[0],centroid[1])    
                dist_list.append(dist)
            idx = np.argmin(dist_list)
            clusters[idx].append(box)
        
        for j in range(para.NUM_CLUSTER):
            centroid = (0,0)
            for k in range(len(clusters[j])):
                centroid = np.add(centroid,clusters[j][k])
            centroid = (centroid[0]/len(clusters[j]),centroid[1]/len(clusters[j]))
            centroids[j] = centroid
                     
            for k in range(len(clusters[j])):
                iou = calculateIoU(0,0,clusters[j][k][0],clusters[j][k][1],0,0,centroid[0],centroid[1])
                avg_iou = avg_iou + iou
            clusters[j] = []
            
        avg_iou = avg_iou / len(boxes)
        if avg_iou >= max_iou:
            max_iou = avg_iou
            optimal_anchor = centroids
            print('Iter = %d/%d, Average IoU = %g, is current optimal anchors.' % (i+1,para.MAX_ITERS,avg_iou))
        else:print('Iter = %d/%d, Average IoU = %g' % (i+1,para.MAX_ITERS,avg_iou))
    
    # save anchors
    with open(os.path.join(para.PATH,'anchor','anchor.txt'),'w') as f:
        optimal_anchor = np.array(optimal_anchor)/para.INPUT_SIZE
        f.write('Width,Height\n')
        for i in range(para.NUM_CLUSTER):
            f.write(str(optimal_anchor[i][0])+',')
            f.write(str(optimal_anchor[i][1])+'\n')
    print('The optimal anchors are: \n',optimal_anchor)       
    return optimal_anchor
        
    
if __name__ == '__main__':
    xmlName = os.path.join(para.PATH,'data','train','train.txt')
    xmlPath = os.path.join(para.PATH,'data','annotation','xml')
    dic = init_centroids(xmlPath,xmlName)
    optimal_anchor = anchor_cluster(dic)
