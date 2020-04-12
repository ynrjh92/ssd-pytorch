import torch
from math import sqrt
from math import ceil
from itertools import product as product

class PriorBox(object):
    def __init__(self, inp_size=(300,300)):
        super(PriorBox, self).__init__()
        self.image_h, self.image_w = inp_size

        # Create feature maps size.
        self.feature_maps = list()
        for i in [8, 16, 32, 64, 128, 300]: 
            self.feature_maps.append([ceil(self.image_h/(i)), ceil(self.image_w/(i))])

        self.scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        self.aspect_ratios = [[2], [2,3], [2,3], [2,3], [2], [2]]

    def forward(self):
        pbox = []
        # Loop each Feature map ([38,19,10,5,3,1])
        for k, feature in enumerate(self.feature_maps):
            fh, fw = feature
            scale = self.scales[k]
            scale_larger = 1 if k+1==len(self.scales) else sqrt(self.scales[k]*self.scales[k+1])
            # Loop each axis
            for y, x in product(range(fh), range(fw)):
                # Unit center x,y
                cx = (x + 0.5) / fw
                cy = (y + 0.5) / fh
                pbox.append([cx, cy, scale, scale])
                pbox.append([cx, cy, scale_larger, scale_larger])
                for ar in self.aspect_ratios[k]:
                    pbox.append([cx, cy, scale * sqrt(ar), scale / sqrt(ar)])
                    pbox.append([cx, cy, scale / sqrt(ar), scale * sqrt(ar)])

        # [num_priors , 4]
        pbox = torch.FloatTensor(pbox)
        pbox.clamp_(min=0, max=1)
        return pbox

if __name__ == '__main__':
    priors = PriorBox((300,300)).forward()
    print(priors)