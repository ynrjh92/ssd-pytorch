import torch
from pprint import PrettyPrinter
from src.utils import jaccard, load_checkpoint, reverse_label_map
from argparser import get_eval_argument, set_cuda_dev
from src.model import SSD
from src.data.datasets import VOCxx

"""
Precision :                           TP
                        ----------------------------
                           TP + FP (all detections)

Recall :                              TP
                        ----------------------------
                              all number of GTs
"""

def collect_results(detections, targets):    
    # detections_all : [N, 7] - [coordinates(4), score, class_label, image_label]
    # targets_all    : [N, 8] - [coordinates(4), class_label, difficulty, image_label, detected or not]
    detections_all = torch.zeros(1,7).cuda()
    targets_all = torch.zeros(1, 8).cuda()
    for i in range(len(detections)):
        # Added a column to track what image it is ([N,6] to [N,7])
        image_track_dim = torch.cat((detections[i],(torch.zeros(detections[i].size(0), 1) + i).cuda()), dim=1)
        detections_all = torch.cat((detections_all, image_track_dim), dim=0)
        # Added a column to keep track of which image it is and whether it has already been detected
        # [N,6] to [N,8]
        image_detected_track_dim = torch.cat((targets[i],(torch.zeros(targets[i].size(0), 1) + i).cuda(), torch.zeros(targets[i].size(0),1).cuda()), dim=1)
        targets_all = torch.cat((targets_all, image_detected_track_dim), dim=0)

    # Remove first garbage row
    detections_all = detections_all[1:, :]
    targets_all = targets_all[1:, :]
    return detections_all, targets_all

def calculate_mAP(detections, targets, args):
    # detections : List of tensors
    # targets : List of tensors
    assert len(detections) == len(targets)
    
    # Collect results for batch length in one tensor.
    # At this time, a column indicating the state required for mAP calculation is added. 
    detections_all, targets_all = collect_results(detections, targets)
    
    # Loop all images
    average_precisions = torch.zeros((args.nclasses-1), dtype=torch.float)
    for c in range(1, args.nclasses):
        # Mask corresponding to a specific class
        detections_cls_mask = detections_all[:, -2].int() == c
        targets_cls_mask = targets_all[:, -4].int() == c

        # Get only the (detections or targets) corresponding to a specific class
        detections_specific_class = detections_all[detections_cls_mask] # [n_detections_specific_class, 7]
        targets_specific_class = targets_all[targets_cls_mask] # [n_targets_specific_class, 8]

        # Sort based on score, then calculate sequentially
        _, sorted_idx = torch.sort(detections_specific_class[:, -3], dim=0, descending=True)
        detections_specific_class = detections_specific_class[sorted_idx]

        TP = torch.zeros((detections_specific_class.size(0)), dtype=torch.float).cuda()
        FP = torch.zeros((detections_specific_class.size(0)), dtype=torch.float).cuda()
        for k, sorted_detection in enumerate(detections_specific_class):
            # Fetch targets of images corresponding to each detections
            targets_image_mask = targets_specific_class[:, -2] == sorted_detection[6] # image_label mask 
            targets_corres_image = targets_specific_class[targets_image_mask]

            # There are no targets
            if targets_corres_image.size(0) == 0:
                FP[k] = 1
                continue

            # Calculate overlap
            overlap = jaccard(targets_corres_image[:, :4], sorted_detection.unsqueeze(0)[0, :4].unsqueeze(0))
            best_target_overlap, best_target_idx = overlap.max(dim=0)

            if best_target_overlap > args.mAP_threshold:
                # To determine whether the target has been detected
                detected_index = targets_image_mask.nonzero()[best_target_idx].item()
                # Performance evaluation is performed only in case of difficulty
                if targets_specific_class[detected_index, -3] == 0:
                    # If the GT has not been detected,
                    # Increase the TP and change the GT to the detected status.
                    if targets_specific_class[detected_index, -1] == 0:
                        targets_specific_class[detected_index, -1] =1
                        TP[k] = 1
                    else:
                        FP[k] = 1
            else:
                FP[k] = 1

        # Accumulated TP and FP for a specific class
        cumul_true_positives = torch.cumsum(TP, dim=0)
        cumul_false_positives = torch.cumsum(FP, dim=0) 
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
        non_diff_examples = (targets_specific_class[:, -3] == 0).sum()
        cumul_recall = cumul_true_positives / non_diff_examples

        # Calculate 11-point interpolated AP
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist() 
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).cuda()
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean() 

    mean_average_precision = average_precisions.mean().item()
    # For pretty printer
    average_precisions = {reverse_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}
    return average_precisions, mean_average_precision


def evaluate(test_loader, model, args):
    # Good formatting when printing the APs for each class and mAP
    pp = PrettyPrinter()

    detections_boxes, target_boxes = list(), list()
    with torch.no_grad():
        # Batches
        for i, (images, targets) in enumerate(test_loader):
            images = images.cuda()
            targets = [t.cuda() for t in targets]

            # Forward propagation
            det_results = model(images)

            # Collect all detection results and targets
            detections_boxes.extend(det_results)
            target_boxes.extend(targets)
            # break
        # Calculate mAP
        APs, mAP = calculate_mAP(detections_boxes, target_boxes, args)

    # Print AP for each class
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)

if __name__ == '__main__':
    # Get eval arguments
    args = get_eval_argument()
    print('Arguments for evaluation : ', args)

    # Set cuda device
    set_cuda_dev(args.ngpu)

    # Load model checkpoint that is to be evaluated
    model = SSD('test', args)
    checkpoint = args.trained_model
    _, model, _ = load_checkpoint(model, args.trained_model_path+checkpoint)
    model = model.cuda()
    # Switch to eval mode
    model.eval()

    # Load test datas
    test_dataset = VOCxx('test', args.dataroot, args.datayears, args.datanames, discard_difficult=args.discard_difficult, use_augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=1, pin_memory=True)

    evaluate(test_loader, model, args)
