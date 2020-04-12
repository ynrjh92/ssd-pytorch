import os
import torch
from src.data.voc_utils import VOC_CLASSES

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
                        '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
# Create dict 
label_map = {k: v + 1 for v, k in enumerate(VOC_CLASSES)}
label_map['background'] = 0
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
# Inverse mapping of key-values
reverse_label_map = {v: k for k, v in label_map.items()}  

# To calculate the jaccard overlap, use the amdegroot method from
# https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounboundingding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)

    # Find left and right box
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    # (x2-x1) * (y2-y1)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter) 
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter) 
    union = area_a + area_b - inter
    return inter / union

def adjust_lr(optimizer, scale=0.1):
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] * scale
    print('LR decay to : ', g['lr']    )
    return optimizer

# Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
# See https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/train.py#L44
def init_optimizer(model, args):
    biases, weights = list(), list()
    for param_name, param in model.named_parameters():
        if param_name.endswith('bias'):
            biases.append(param)
        else:
            weights.append(param)
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * args.lr},
                                                                {'params': weights}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return optimizer

def load_checkpoint(model, checkpoint):
    checkpoint = torch.load(checkpoint)
    resume_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer = checkpoint['optimizer']
    print('Loaded checkpoint from ' + str(resume_epoch-1) + ' epoch.')
    return resume_epoch, model, optimizer

def save_checkpoint(epoch, model, optimizer, save_name):
    state = {'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer}
    filename = os.getcwd()+'/results/weights/'+str(save_name)+'_'+str(epoch)+'epoch.pth'
    torch.save(state, filename)

def clip_gradient(params, grad_norm):
    torch.nn.utils.clip_grad_norm_(params, grad_norm)

def plot_losses(loss, label, name):
    import matplotlib.pyplot as plt
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(loss, label=label)
    plt.savefig(os.getcwd()+'/results/loss/' + str(label)+'_'+str(name) + '_loss.png')
    plt.legend(loc='upper right')
    plt.close()

if __name__ == '__main__':
    print()