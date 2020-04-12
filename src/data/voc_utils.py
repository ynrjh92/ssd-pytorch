import os
import numpy as np
import torch
# 21 classes include bkg.
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def parse_annotation(anno_file):
    import xml.etree.ElementTree as ET 
    anno_tree = ET.parse(anno_file).getroot()
    annotations = list()
    bbox_form = ['xmin', 'ymin', 'xmax', 'ymax']
    for obj in anno_tree.findall('object'):
        boxes = [obj.find('bndbox').find(elem).text for elem in bbox_form]
        label = [obj.find('name').text]
        boxes += [VOC_CLASSES.index(label[0])+1] # get class label.
        diff = [obj.find('difficult').text]
        boxes += diff
        annotations.append(boxes)
    annotations = torch.from_numpy(np.array(annotations, dtype=np.float32))
    return annotations


def find_overlapped_elemenets(datadir, dataset_file):
    ext = ['.jpg', '.xml']
    img_ext, anno_ext = list(), list()
    with open(dataset_file) as f:
        datas = [line.rstrip() for line in f] # remove '\n'

    img_ext = [data + ext[0] for data in datas]
    anno_ext = [data + ext[1] for data in datas]
    img_total = sorted(os.listdir(datadir+ '/JPEGImages/'))
    anno_total = sorted(os.listdir(datadir+ '/Annotations/'))
    return list(sorted(set(img_ext) & set(img_total))), list(sorted(set(anno_ext) & set(anno_total)))

def parse_data(datadirs, datanames, images=[], annotations=[]):
    if ',' in datadirs[0]:
        datadirs = datadirs[0].split(',')
    
    for datadir in datadirs:
        dataset_file = datadir + '/ImageSets/Main/' + datanames    
        # Get only overlapped elemenets.
        imgs, rois = find_overlapped_elemenets(datadir, dataset_file)
        # Convert each name to include absolute path.
        imgs = [datadir+'/JPEGImages/'+img for img in imgs]
        rois = [datadir+'/Annotations/'+roi for roi in rois]
        images += imgs; annotations += rois
    return images, annotations
