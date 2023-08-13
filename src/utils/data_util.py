import torch
import cv2 as cv
import numpy as np
from einops import reduce

def TensorToCV(image_tensor: torch.Tensor): # B,C,H,W
    # tens = image_tensor * 255 / reduce(image_tensor, 'b c h w -> b', 'max')
    tens = image_tensor * 255 / image_tensor.max(1, keepdim=True)[0].max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    image_cv = tens.detach().cpu().numpy().astype(np.uint8).transpose(0,2,3,1)    # b c h w -> b h w c
    images = [cv.cvtColor(image, cv.COLOR_BGR2RGB) for image in image_cv]
    return images