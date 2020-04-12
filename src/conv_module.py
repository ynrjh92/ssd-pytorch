import torch.nn as nn

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

class extra_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride, pad=0):
        super(extra_conv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=stride, padding=pad), nn.ReLU(inplace=True)
                )
        init_weights(self.conv.modules())
    def forward(self, x):
        return self.conv(x)

class pred_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super(pred_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        init_weights(self.conv.modules())
    def forward(self, x):
        return self.conv(x)