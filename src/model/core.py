import torch
from einops import rearrange
import torch.nn.functional as F


def quadratic_flow(F0n1: torch.Tensor, F01: torch.Tensor, gamma: float, tau: float) -> torch.Tensor:
    """solve the quadratic motion matrix and predict the correction feild.
    Args:
        F0n1 (torch.Tensor): flow 0 -> -1
        F01 (torch.Tensor): flow 0 -> 1
        gamma (float): the readout reatio
        tau (float): the timestamp warping to
    Returns:
        torch.Tensor: the correction feild to tau.
    """
    h, w = F0n1.shape[1:3]
    t0n1 = -1 + gamma / h * F0n1[:, :, :, 1]
    t01 = 1 + gamma / h * F01[:, :, :, 1]

    # solve the quadratic motion matrix
    A = rearrange(
        torch.stack([t0n1, 0.5 * t0n1**2, t01, 0.5 * t01**2], dim=-1),
        "b h w (m n) -> b h w m n",
        m=2,
        n=2,
    )
    B = torch.stack([F0n1, F01], dim=-2)
    M = torch.linalg.solve(A, B)

    # predict the correction feild
    grid_y, _ = torch.meshgrid(
        torch.arange(0, h, device=F0n1.device, requires_grad=False),
        torch.arange(0, w, device=F0n1.device, requires_grad=False),
    )
    t0tau = tau - gamma / h * grid_y

    Atau = rearrange(torch.stack([t0tau, 0.5 * t0tau**2], dim=-1), "h w m -> h w 1 m")
    F0tau = rearrange(Atau @ M, "b h w 1 n -> b h w n")

    return F0tau


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def feats_sampling(
    x,
    flow,
    interpolation="bilinear",
    padding_mode="zeros",
    align_corners=True,
):
    """return warped images with flows in shape(B, C, H, W)
    Args:
        x: shape(B, C, H, W)
        flow: shape(B, H, W, 2)
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f"The spatial sizes of input ({x.size()[-2:]}) and " f"flow ({flow.size()[1:3]}) are not the same.")
    h, w = x.shape[-2:]

    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # * (h, w, 2)
    grid.requires_grad = False
    grid_flow = grid + flow

    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return output
