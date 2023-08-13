import torch
import torch.nn as nn
import torch.nn.functional as F
from package_core.losses import PerceptualLoss, VariationLoss
from torch.nn.modules.loss import _Loss


def l1_loss(output, target):
    return F.l1_loss(output, target)


# * Zhong, JCD, CVPR21, loss = 10 * L1_Charbonnier_loss_color | 1 * Perceptual | 0.1 * Variation
class L1_Charbonnier_loss_color(_Loss):
    def __init__(self):
        super(L1_Charbonnier_loss_color, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


def Perceptual():
    return PerceptualLoss(loss=nn.L1Loss())


def Variation():
    return VariationLoss(nc=2)


class Color_consistency_loss(_Loss):
    def __init__(self):
        super(Color_consistency_loss, self).__init__()

    def forward(self, X, Y):  # B3HW
        return torch.mean(F.l1_loss(X[:, 0], X[:, 1]) + F.l1_loss(X[:, 0], X[:, 2]) + F.l1_loss(X[:, 1], X[:, 2])) / 3


class Loss(nn.Module):
    def __init__(self, losses, ratios, auto=False):
        """get loss list, ratio, and instance them.
        Args:
            losses (List): str of loss function.
            ratios (List): factor of loss function
            auto (bool): loss = loss1 + loss2 / loss2.detach() + loss3 loss3.detach()

        e.g:
        loss:
            _target_: src.model.loss_dev.Loss
            losses: ['L1_Charbonnier_loss_color', 'Perceptual']
            ratios: [1, 0.1]
            auto: True
        """
        super(Loss, self).__init__()
        self.names = losses
        self.ratios = ratios
        self.losses = list(eval("{}()".format(loss)) for loss in losses)
        self.auto = auto
        print(self.losses)

    def forward(self, output, target, flows=None):
        if len(output.shape) == 5:
            b, n, c, h, w = output.shape
            output = output.reshape(b * n, c, h, w)
            target = target.reshape(b * n, c, h, w)

        ret = {"total": torch.tensor(0.0, requires_grad=True).cuda().float()}

        for name, loss in zip(self.names, self.losses):
            # import pdb; pdb.set_trace()
            if name == "Variation":
                ret[name] = torch.tensor(0.0, requires_grad=True).cuda().float()
                for flow in flows:
                    ret[name] += loss(flow, mean=True)
            elif name == "Perceptual":
                ret[name] = loss.get_loss(output, target)
            else:
                ret[name] = loss(output, target)

        for name, ratio in zip(self.names[1:], self.ratios[1:]):
            if self.auto:  # norm
                ret[name] /= (ret[name] / ret[self.names[0]]).detach()
            ret[name] *= ratio
            ret["total"] += ret[name]

        ret[self.names[0]] *= self.ratios[0]
        ret["total"] += ret[self.names[0]]

        return ret
