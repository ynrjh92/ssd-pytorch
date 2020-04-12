import torch
import torch.nn as nn
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self,in_ch=512, scale=20):
        super(L2Norm,self).__init__()
        self.in_ch = in_ch
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(1, self.in_ch, 1, 1))
        self.init_params()

    def init_params(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x,norm)
        out = x * self.weight
        return out

if __name__ == '__main__':
    x = torch.randn([1, 512, 38, 38])
    l2norm = L2Norm(512, 20)(x)