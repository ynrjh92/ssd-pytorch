import os
import argparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def set_cuda_dev(ngpu):
    import torch
    import torch.backends.cudnn as cudnn
    torch.cuda.set_device(ngpu)
    cudnn.benchmark = True

def get_train_argument():
    parser = argparse.ArgumentParser(description='PyTorch-based SSD implementation')
    parser.add_argument('--ngpu', default=0, type=int, help='(int) Select which GPU to use')

    # Data parameters
    parser.add_argument('--discard-difficult', default=False, type=str2bool, help='(bool) Decide whether to use the difficulty example or not')
    parser.add_argument('--dataroot', default=['/media/dataset/VOC2007','/media/dataset/VOC2012'], nargs='*', help='(list) Set the root path of the VOC dataset')
    parser.add_argument('--datayears', default='07+12', type=str, help='(str) Determine the year of the VOC dataset (07 / 12 / 07+12)')
    parser.add_argument('--datanames', default='trainval.txt', type=str, help='(str) Determine the datanames of the VOC dataset (e.g. trainval.txt)')    
    parser.add_argument('--use-augment', default=True, type=str2bool, help='(bool) Decide whether to use data augmentation or not')
    parser.add_argument('--batch-size', default=8, type=int, help='(int) Batch size')

    # Model parameters
    parser.add_argument('--nclasses', default=21, type=int, help='(int) The number of classes')
    parser.add_argument('--resume', default=False, type=str2bool, help='(bool) Determine whether resume mode or not')
    parser.add_argument('--pretrained-model', default=os.getcwd()+'/results/weights/pretrained/vgg16_converted.pth', type=str, help='(str) Where is the pre-trained vgg16 weight model')
    parser.add_argument('--trained-model', default=None, type=str, help='(str) Where the trained model is')
    parser.add_argument('--model-save-name', default=None, type=str, help='(str) The name you want to save the trained model to')
    
    # Learning parameters
    parser.add_argument('--iterations', default=120000, type=int, help='(int) Determine number of iterations to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='(float) Set the start learning rate')  
    parser.add_argument('--lr-decay', default='80000,100000', type=str, help='(str) Determine when learning rate decay after these iterations')
    parser.add_argument('--momentum', default=0.9, type=float, help='(float) Set momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='(float) Set weight decay')
    parser.add_argument('--clip-grad', default=0., type=float, help='(float) Clip to prevent gradient exploiding')
    parser.add_argument('--overlap-threshold', default=0.5, type=float, help='(float) Overlap threshold that is the basis for positive and negative judgment')
    parser.add_argument('--negpos-ratio', default=3, type=int, help='(int) Ratio of negative and positive samples')
    parser.add_argument('--alpha', default=1.0, type=float, help='(float) Weight for regression loss')

    args = parser.parse_args()
    return args

def get_eval_argument():
    parser = argparse.ArgumentParser(description='PyTorch-based SSD implementation')
    parser.add_argument('--ngpu', default=0, type=int, help='(int) Select which GPU to use')
    # Data parameters
    parser.add_argument('--discard-difficult', default=False, type=str2bool, help='(bool) Decide whether to use the difficulty example or not')
    parser.add_argument('--dataroot', default=['/media/dataset/VOC2007'], nargs='*', help='(list) Set the root path of the VOC dataset')
    parser.add_argument('--datayears', default='07+12', type=str, help='(str) Determine the year of the VOC dataset (07 / 12 / 07+12)')
    parser.add_argument('--datanames', default='test.txt', type=str, help='(str) Determine the datanames of the VOC dataset (e.g. test.txt)')    
    parser.add_argument('--batch-size', default=1, type=int, help='(int) Batch size')

    # Model parameters
    parser.add_argument('--nclasses', default=21, type=int, help='(int) The number of classes')
    parser.add_argument('--pretrained-model', default=os.getcwd()+'/results/weights/pretrained/vgg16_converted.pth', type=str, help='(str) Where is the pre-trained vgg16 weight model')
    parser.add_argument('--trained-model', default=None, type=str, help='(bool) Where the trained model is')
    parser.add_argument('--min-score', default=0.01, type=float, help='(float) Threshold score to judge whether object is detected')
    parser.add_argument('--top-k', default=200, type=int, help='(int) To get only the top_k scores with high scores among multiple detections')
    parser.add_argument('--nms-overlap', default=0.45, type=float, help='(float) Threshold used in nms algorithm')
    parser.add_argument('--mAP-threshold', default=0.5, type=float, help='(float) Overlap criteria to judge TP and FP')

    # Result parameters
    parser.add_argument('--trained-model-path', default=os.getcwd()+'/results/weights/', type=str, help='(str) Path where the trained model is saved')
    
    args = parser.parse_args()
    return args

def get_infer_argument():
    parser = argparse.ArgumentParser(description='PyTorch-based SSD implementation')
    parser.add_argument('--ngpu', default=0, type=int, help='(int) Select which GPU to use')
    # Data parameters
    parser.add_argument('--discard-difficult', default=False, type=str2bool, help='(bool) Decide whether to use the difficulty example or not')
    parser.add_argument('--dataroot', default=['/media/dataset/VOC2007'], nargs='*', help='(list) Set the root path of the VOC dataset')
    parser.add_argument('--datayears', default='07+12', type=str, help='(str) Determine the year of the VOC dataset (07 / 12 / 07+12)')
    parser.add_argument('--datanames', default='test.txt', type=str, help='(str) Determine the datanames of the VOC dataset (e.g. test.txt)')    
    parser.add_argument('--batch-size', default=1, type=int, help='(int) Batch size')

    # Model parameters
    parser.add_argument('--nclasses', default=21, type=int, help='(int) The number of classes')
    parser.add_argument('--pretrained-model', default=os.getcwd()+'/results/weights/pretrained/vgg16_converted.pth', type=str, help='(str) Where is the pre-trained vgg16 weight model')
    parser.add_argument('--trained-model', default=None, type=str, help='(bool) Where the trained model is')
    parser.add_argument('--min-score', default=0.2, type=float, help='(float) Threshold score to judge whether object is detected')
    parser.add_argument('--top-k', default=200, type=int, help='(int) To get only the top_k scores with high scores among multiple detections')
    parser.add_argument('--nms-overlap', default=0.45, type=float, help='(float) Threshold used in nms algorithm')
    
    # Result parameters
    parser.add_argument('--test-image', default='/media/dataset/VOC2007/JPEGImages/000001.jpg', type=str, help='(str) Test image path and name to confirm detection results')
    parser.add_argument('--image-save-path', default=os.getcwd()+'/results/images/', type=str, help='(str) Path to save the resulting image')
    parser.add_argument('--trained-model-path', default=os.getcwd()+'/results/weights/', type=str, help='(str) Path where the trained model is saved')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print()