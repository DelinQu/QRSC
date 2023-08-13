import torch
from src.lib.lpips import lpips
from package_core import metrics

def psnr(output, target, **args):
    with torch.no_grad():
        return metrics.PSNR(output, target, mask = None)

def psnr_mask(output, target, **args):
    with torch.no_grad():
        return metrics.PSNR(output, target, args['mask'])

def ssim(output, target, **args):
    with torch.no_grad():
        return metrics.SSIM(output, target).item()

LPIPS = lpips.LPIPS(net="alex")
def lpips(output, target, **args):
    with torch.no_grad():
        return torch.mean(LPIPS(output, target)).item()