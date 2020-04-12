import torch
import torch.nn as nn
from src.utils import jaccard

def decode(encoded_preds, priors_cxcy):
    preds_cxcy = encoded_preds[:,:2]*priors_cxcy[:,2:]*0.1 + priors_cxcy[:,:2]
    preds_wh = torch.exp(encoded_preds[:,2:]*0.2) * priors_cxcy[:, 2:]
    preds_center = torch.cat((preds_cxcy, preds_wh), dim=1)
    preds_boundary = center_to_boundary(preds_center)
    return preds_boundary
    
def encode(cxcy, priors_cxcy):
    g_cxcy = (cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:]*0.1)
    g_wh = cxcy[:, 2:] / priors_cxcy[:, 2:]
    g_wh = torch.log(g_wh+1e-10)*5
    return torch.cat((g_cxcy, g_wh), dim=1)

def boudnary_to_center(box):
    cxcy = (box[:, :2] + box[:, 2:])/2
    wh = box[:, 2:] - box[:, :2]
    center = torch.cat((cxcy,wh), dim=1)
    return center

def center_to_boundary(box):
    x1y1 = box[:, :2] - (box[:, 2:]/2)
    x2y2 = box[:, :2] + (box[:, 2:]/2)
    boundary = torch.cat((x1y1,x2y2), dim=1)
    return boundary

# The steps for calculating the loss are well described below and are referenced
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/model.py#L532
class MultiBoxLoss(nn.Module):
    def __init__(self, priors, overlap_threhsold, negpos_ratio, alpha):
        super(MultiBoxLoss, self).__init__()
        self.prior_boxes = priors
        self.prior_boxes_boundary = center_to_boundary(self.prior_boxes)
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        # parameters
        self.overlap_threshold = overlap_threhsold
        self.negpos_ratio = negpos_ratio
        self.alpha = alpha
        
    def forward(self, preds, targets):
        pred_locs, pred_confs = preds
        nbatch, npriors = pred_locs.size()[:2]
        nclasses = pred_confs.size(2)
        encoded_target_locs = torch.zeros((nbatch, npriors, 4), dtype=torch.float).cuda() # [nbatch,npriors,4]
        encoded_target_confs = torch.zeros((nbatch, npriors), dtype=torch.long).cuda() # [nbatch,npriors]

        # Loop batches
        for k in range(nbatch):
            overlap = jaccard(targets[k][:,:-2], self.prior_boxes_boundary) # [n_targets, n_priors]
            # Calculation of mutual overlap between prior boxes and targets
            best_gt_overlap, best_gt_idx = overlap.max(dim=0) # [n_priors]
            _, best_prior_idx = overlap.max(dim=1) # [n_targets]

            # Based on targets, elements (prior box) with max overlap are first secured (by setting the corresponding target label and set overlap to 1).
            best_gt_idx[best_prior_idx] = torch.LongTensor(range(targets[k].size(0))).cuda() # Assign each gt label
            best_gt_overlap[best_prior_idx] = 1.
            
            # Prior box doesnot have label information, so target label information is assigned.
            best_gt_label = targets[k][:,-2][best_gt_idx]
            
            # For all prior boxes, anything below the threshold is set to negative priors (not used for regression loss).
            best_gt_label[best_gt_overlap<self.overlap_threshold]  = 0            
            encoded_target_confs[k] = best_gt_label

            # To create a target for learning, we encode the prior boxes and targets.
            # At this time, targets are those with (overlap>threshold) based on prior box.
            # Since prior boxes with max overlap based on GT were secured in the previous step, it satisfies the matching strategy of SSD paper.
            encoded_target_locs[k] = encode(boudnary_to_center(targets[k][:,:-2][best_gt_idx]), self.prior_boxes)

        # Regression loss only for positive samples
        pos_priors_idx = encoded_target_confs > 0
        loc_loss = self.smooth_l1(pred_locs[pos_priors_idx], encoded_target_locs[pos_priors_idx])

        ### Classification loss requires learning both positive and hard negative samples.        
        n_positives = pos_priors_idx.sum(dim=1) # [batch]
        n_negatives = n_positives*self.negpos_ratio

        # We want to calculate the loss for all prior boxes and find a hard negative sample based on the calculated loss.
        conf_loss_all_priors = self.cross_entropy(pred_confs.view(-1, nclasses), encoded_target_confs.view(-1)) # [batch*nprior, nclasses]
        conf_loss_all_priors = conf_loss_all_priors.view(nbatch, npriors)

        # For positive sample loss
        conf_loss_pos = conf_loss_all_priors[pos_priors_idx]

        # Exclude positive priors from top_k (to clearly prevent duplication).
        conf_loss_neg = conf_loss_all_priors.clone()        
        conf_loss_neg[pos_priors_idx] = 0.

        # After creating a sequential number from 0 to the number of priors,
        # then get it from the front by the number of negative samples.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        sequential_ranks = torch.LongTensor(range(npriors)).unsqueeze(0).expand_as(conf_loss_neg).cuda()        
        hard_negatives_mask = sequential_ranks < n_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives_mask]
        
        # TOTAL LOSS
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float() 

        return conf_loss, self.alpha*loc_loss        

if __name__ == '__main__':
    print()