# SSD: Single Shot Multibox Detector, PyTorch  

This is **PyTorch** implementation of SSD [Paper](https://arxiv.org/pdf/1512.02325.pdf).  
The representative implementations based on **PyTorch** are [amdegroot](https://github.com/amdegroot/ssd.pytorch) and [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection), and both repositories are referenced as needed.  
See [Comparison of both repositories](#Comparisons) for differences between the two repositories.  

# Getting started

#### Sample results
<img src=./results/images/p_000001.jpg width="250" height="250">  <img src=./results/images/p_000002.jpg width="250" height="250">  <img src=./results/images/p_000004.jpg width="250" height="250">

## Table of contents  
- [Installation](#Installation)  
- [Datasets](#Datasets)  
- [Train](#Train)  
- [Inference](#Inference)  
- [Evaluation](#Evaluation)  
- [Arguments](#Arguments)  
- [Performance](#Performance)  

## Installation  
- python>=3.6
- PyTorch>=0.4
- PIL>=6.2.1
- torchvision>=0.5.0
- matplotlib>=3.1.0

## Datasets  
- Download all of the [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) and [2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) datasets.  
For consistency, the directory structure of both datasets must be matched as follows.

- Data root directory  
├── **VOC 2007**  
│   └── Annotations  
│   └── ImageSets  
│   └── JPEGImages  
│   └── SegmentationClass    
│   └── SegmentationObject  
├── **VOC 2012**  
│   └── Annotations  
│   └── ImageSets  
│   └── JPEGImages  
│   └── SegmentationClass  
│   └── SegmentationObject  

## Train
You must manually modify the root path of the VOC dataset (dataroot) in ```argparser.py```.  
Then, you can train as below.
```  
python train.py  
```
Otherwise, there is a way to give it as argument as below (To differentiate between two datasets, use only ",").  
```
python train.py --dataroot=/media/dataset/VOC2007,/media/dataset/VOC2012  
```
The trained model weights and loss graphs created during the training are stored in the ```./results/weights/``` and ```./results/loss``` folders, respectively.  

## Inference
```
python detect_image.py --trained-model="trained_model_name.pth" --test-image="test_image_name_with_absolute_path" 
```
The result images are saved in the ```./results/images/``` folder with the ```p_``` prefix in front of the original image name.  

## Evaluation
```
python calc_mAP.py --trained-model="trained_model_name.pth"
```  

## Arguments
- Many parameters are required to train and evaluate the object detector.  
For this, the arguments used for training and evaluation are defined separately in ```argparser.py```.  
By using this, you can easily use and tune the parameters.  
- The above-described execution examples ([Train](#Train), [Inference](#Inference), [Evaluation](#Evaluation)) are used only the minimum parameters required for operation, and all of the rest parameters are set as default.  

## Performance  

|VOC 2007 test | Achieved | Reported paper |  Trained model |  
| :--------: | :------: | :--------: | :--------: |  
| mAP (%)  | 76.5 | 77.2 | [download](https://drive.google.com/file/d/1wZVIJ5KaJlz9CTyrkXRwIek_VWNYBz6W/view?usp=sharing) |  


## Comparisons  
- The weights of the **pre-trained VGG16** used in the **amdegroot** repository and the weights of the **sgrvinod** directly converting the FC layer weights to the weights of the convolution layer are different.  
  
- When **sgrvinod** initializing the optimizer according to the [original caffe repo](https://github.com/weiliu89/caffe/tree/ssd) the **learning rate for bias terms was set to twice**, but in the **amdegroot** implementation, there is no such information.  

- There is a slight difference in the **size of the prior boxes** made in each repository.

- There is some difference in the **calculated loss** by each repository (There is a difference in how the loss is calculated.).  

- **sgrvinod's** repository is written for easy understanding and seems to follow the official paper more strictly, but the implementation of **amdegroot** is more optimized to PyTorch library.  
(Especially, it is prominent in time-consuming **Non-maximum suppression (NMS)** step)

- Therefore, I combined the two repository implementations as needed.


#### Feedback on anything wrong is always welcome!
