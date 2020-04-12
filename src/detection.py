import torch
import torch.nn as nn
from src.multibox_loss import decode

# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap, top_k):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class pred scores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    # Initialize tensor
    keep = scores.new(scores.size(0)).zero_().long()

    # Exception handle
    if boxes.numel() == 0:  # numel : number of elements ignoring dimensions
        return keep, 0

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    # Sort in ascending order
    score, idx = scores.sort(0)

    # Get top_k elements from rightmost idx (element with the highest score of top_k)
    idx = idx[-top_k:]

    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        # index of current highest score val
        i = idx[-1]

        # Keep highest score index
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break

        # Remove kept element from view
        idx = idx[:-1]

        # Get remain boxes is loaded in order of highest scores
        # For Compare with highest score box and all the other boxes
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # Store element-wise max with next highest score
        # For calculate intersection
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, min=0, max=x2[i])
        yy2 = torch.clamp(yy2, min=0, max=y2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = xx2 - xx1
        h = yy2 - yy1

        # Check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h

        
        # area[i] : highest score box
        # rem_areas : all other boxes
        # inter : intersection with (area[i] and rem_areas)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # Store result in iou

        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

class Detect(nn.Module):
    """
        Decode the detection results (offset) obtained from the network and
        Remove duplicated detection by using NMS algorithm for each class
    """
    def __init__(self, priors, min_score, nms_overlap, nclasses, top_k):
        super(Detect, self).__init__()
        self.priors = priors
        self.top_k = top_k
        self.nclasses = nclasses
        self.min_score = min_score
        self.nms_overlap = nms_overlap
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, preds):
        # pred_locs : [nbatch, npriors, 4]
        # pred_scores : [nbatch, npriors, nclasses]
        pred_locs, pred_scores = preds
        pred_scores = self.softmax(pred_scores)

        batch_results = list()
        for batch in range(pred_locs.size(0)):
            # Decode predict boxes
            decoded_boxes = decode(pred_locs[batch], self.priors.cuda())

            # Perform NMS for each class.
            scores_each_class = pred_scores[batch].clone()
            cls_indep_results = list()
            for cls_idx in range(1, self.nclasses):
                # Filter out mask above minimum score threshold.
                rm_mask = scores_each_class[:, cls_idx].gt(self.min_score)
                
                # Remove boxes and scores below the min_score threshold
                scores = scores_each_class[:, cls_idx][rm_mask]
                boxes = decoded_boxes[rm_mask, :]

                # Do NMS
                ids, cnt = nms(boxes, scores, self.nms_overlap, self.top_k)
                scores = scores[ids[:cnt]]  # [N]
                boxes = boxes[ids[:cnt]]    # [N,4]

                # Save results after nms (save all, regardless of the target class label)
                labels = (torch.zeros(1).expand_as(scores) + cls_idx).cuda()
                cls_indep_results.append(torch.cat((boxes, scores.unsqueeze(1), labels.unsqueeze(1)), dim=1))

            # Select only top_k.
            # [N, 6 (=4 location, 1 score, 1 cls_label)] : 
            cls_indep_results = torch.cat(cls_indep_results, dim=0)
            if cls_indep_results.size(0) > self.top_k:
                _, sort_idx = cls_indep_results[:, 4].sort(dim=0, descending=True)
                cls_indep_results = cls_indep_results[sort_idx][:self.top_k]

            batch_results.append(cls_indep_results)
        return batch_results

if __name__ == '__main__':
    print()