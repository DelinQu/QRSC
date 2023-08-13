import random
from torchvision import transforms
import torchvision.transforms.functional as F

class RandomCrop(object):
    def __init__(
        self,
        crop_H=448,
        crop_W=256,
        tsfm=transforms.Compose([transforms.ToTensor()]),
    ):
        self.crop_H, self.crop_W = crop_H, crop_W
        self.tsfm = tsfm

    def __call__(self, size, **args):
        W, H = size  # PIL size return W, H
        random.seed(random.randint(0, 2**32))
        top, left = random.randint(0, H - self.crop_H), random.randint(
            0, W - self.crop_W
        )
        ret = list()
        for _, value in args.items():
            if isinstance(value, list):
                for i, img in enumerate(value):
                    value[i] = self.tsfm(F.crop(img, top, left, self.crop_H, self.crop_W))
            else:
                value = self.tsfm(F.crop(value, top, left, self.crop_H, self.crop_W))
            ret.append(value)

        return tuple(ret)

    def __repr__(self):
        return self.__class__.__name__
    
class ToTensor(object):
    def __init__(
        self,
        tsfm=transforms.Compose([transforms.ToTensor()]),
    ):
        self.tsfm = tsfm

    def __call__(self, size, **args):
        ret = list()
        for _, value in args.items():
            if isinstance(value, list):
                for i, img in enumerate(value):
                    value[i] = self.tsfm(img)
            else:
                value = self.tsfm(value)
            ret.append(value)
        return tuple(ret)

    def __repr__(self):
        return self.__class__.__name__
    