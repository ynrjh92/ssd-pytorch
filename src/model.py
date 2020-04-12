import torch
import torch.nn as nn 
from src.conv_module import extra_conv, pred_conv
from src.prior_box import PriorBox
from src.detection import Detect
from src.l2norm import L2Norm

class SSD(nn.Module):
    def __init__(self, phase, args):
        super(SSD, self).__init__()
        # Define parameters
        self.phase = phase
        self.args = args
        self.num_dbox = [4, 6, 6, 6, 4, 4]
        self.nclasses = args.nclasses

        # Backbone vgg networks
        self.vgg = nn.ModuleList(get_vgg16(args)) # up to conv7

        # L2 norm
        self.l2norm = L2Norm(in_ch=512, scale=20)

        # Extra layers (in_ch, out_ch, ch, stride)
        self.extra1 = extra_conv(1024, 256, 512, 2, 1)  # conv8 
        self.extra2 = extra_conv(512, 128, 256, 2, 1)    # conv9
        self.extra3 = extra_conv(256, 128, 256, 1)        # conv10
        self.extra4 = extra_conv(256, 128, 256, 1)        # conv11

        # Multi-scale detection branches (in_ch, out_ch, kernel_size, padding)
        self.loc_pred_convs = nn.ModuleList([
            pred_conv(512, self.num_dbox[0]*4, 3, 1),
            pred_conv(1024, self.num_dbox[1]*4, 3, 1),
            pred_conv(512, self.num_dbox[2]*4, 3, 1),
            pred_conv(256, self.num_dbox[3]*4, 3, 1),
            pred_conv(256, self.num_dbox[4]*4, 3, 1),
            pred_conv(256, self.num_dbox[5]*4, 3, 1)])
        self.cls_pred_convs = nn.ModuleList([
            pred_conv(512, self.num_dbox[0]*self.nclasses, 3, 1),
            pred_conv(1024, self.num_dbox[1]*self.nclasses, 3, 1),
            pred_conv(512, self.num_dbox[2]*self.nclasses, 3, 1),
            pred_conv(256, self.num_dbox[3]*self.nclasses, 3, 1),
            pred_conv(256, self.num_dbox[4]*self.nclasses, 3, 1),
            pred_conv(256, self.num_dbox[5]*self.nclasses, 3, 1)])

        # Create prior boxes only in inference mode
        if self.phase != 'train':
            self.prior_boxes = PriorBox().forward()
            self.detect = Detect(self.prior_boxes,self.args.min_score, self.args.nms_overlap, self.nclasses, self.args.top_k)

    def forward(self, x):
        det_fmaps, loc_preds, cls_preds = [], [], []
        
        # Up to conv4_3
        for k in range(23):
            x = self.vgg[k](x)
        x = self.l2norm(x)
        det_fmaps.append(x)

        # Up to backbone extras networks
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        det_fmaps.append(x)

        # Forwarding to extra layers
        x = self.extra1(x)
        det_fmaps.append(x)
        x = self.extra2(x)
        det_fmaps.append(x)
        x = self.extra3(x)
        det_fmaps.append(x)
        x = self.extra4(x)
        det_fmaps.append(x)

        # Multi-scale detection branches
        for k, feature_map in enumerate(det_fmaps):
            loc_preds.append(self.loc_pred_convs[k](feature_map).permute(0,2,3,1).contiguous().view(x.size(0), -1, 4))
            cls_preds.append(self.cls_pred_convs[k](feature_map).permute(0,2,3,1).contiguous().view(x.size(0), -1, self.nclasses))
            
        # Aggregate detection results
        locs = torch.cat(loc_preds, dim=1)
        scores = torch.cat(cls_preds, dim=1)
        
        output = (locs, scores)
        if self.phase != 'train':
            output = self.detect(output)            
        return output

def get_vgg16(args):
    import torchvision.models as models
    vgg16 = models.vgg16(pretrained=True).features[:-1]
    # "ceil_mode=True" for matching 38x38 feature map size
    vgg16[16] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    vgg16 = list(vgg16.children())
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    # In order to convert the FC6 and FC7 layers of the VGG-16 to convolutional layers, we need to convert the weights
    # See the link on the right for how to convert (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/model.py#L89)
    # I have already saved the converted weights in the "results/weights/pretrained/" directory under the name "vgg16_converted.pth" using above method.
    vgg16_fc2conv = torch.load(args.pretrained_model)
    conv6.weight.data = vgg16_fc2conv['conv6.weight']; conv6.bias.data = vgg16_fc2conv['conv6.bias']
    conv7.weight.data = vgg16_fc2conv['conv7.weight']; conv7.bias.data = vgg16_fc2conv['conv7.bias']
    vgg16 += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return vgg16

if __name__ == '__main__':
    print()