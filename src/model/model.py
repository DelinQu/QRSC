import torch
import torch.nn as nn
import torch.nn.functional as F
from mmflow.apis import inference_model, init_model
from src.utils.data_util import TensorToCV
from src.model.core import quadratic_flow, feats_sampling
from src.model.Sep_STS_Encoder import ResBlock, SepSTS_Encoder
from einops import rearrange
from src.lib.cupy_module import adacof


def joinTensors(X1, X2, type="concat"):
    if type == "concat":
        return torch.cat([X1, X2], dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class upSplit(nn.Module):
    """
    Applies a 3D transposed convolution operator over an input image composed of several input planes.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upconv = nn.ModuleList(
            [
                nn.ConvTranspose3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=(3, 3, 3),
                    stride=(1, 2, 2),
                    padding=1,
                ),
            ]
        )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x, output_size):
        x = self.upconv[0](x, output_size=output_size)
        return x


class Conv_3d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        batchnorm=False,
    ):
        super().__init__()
        self.conv = [
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
        ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)


class Conv_2d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        batchnorm=False,
    ):
        super().__init__()
        self.conv = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        ]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class MySequential(nn.Sequential):
    def forward(self, input, output_size):
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):
                input = module(input, output_size)
            else:
                input = module(input)
        return input


class RSAdaCof(nn.Module):
    def __init__(self, n_inputs, nf, ks, dilation, norm_weight=True):
        super(RSAdaCof, self).__init__()

        # predict alpha and beta
        def Subnet_offset(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
            )

        # predict W
        def Subnet_weight(ks):
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(ks, ks, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                nn.Softmax(1) if norm_weight else nn.Identity(),
            )

        # predict Mask
        def Subnet_occlusion():
            return MySequential(
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.ConvTranspose2d(nf, nf, kernel_size=3, stride=2, padding=1),
                torch.nn.Conv2d(
                    in_channels=nf,
                    out_channels=n_inputs,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.Softmax(dim=1),
            )

        self.n_inputs = n_inputs
        self.kernel_size = ks
        self.kernel_pad = int(((ks - 1) * dilation) / 2.0)
        self.dilation = dilation

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

        self.ModuleWeight = Subnet_weight(ks**2)
        self.ModuleAlpha = Subnet_offset(ks**2)
        self.ModuleBeta = Subnet_offset(ks**2)
        self.moduleOcclusion = Subnet_occlusion()

        self.feature_fuse = Conv_2d(nf * n_inputs, nf, kernel_size=1, stride=1, batchnorm=False, bias=True)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, fea, frames, output_size, delta, tau):
        H, W = output_size
        # fea = BDCHW
        occ = torch.cat(torch.unbind(fea, 1), 1)  # BDCHW => (BCHW, ..... n_input) => B(D*C)HW
        occ = self.lrelu(self.feature_fuse(occ))  # BCHW
        Occlusion = self.moduleOcclusion(occ, (H, W))  #

        B, C, T, cur_H, cur_W = fea.shape
        fea = fea.transpose(1, 2).reshape(B * T, C, cur_H, cur_W)
        weights = self.ModuleWeight(fea, (H, W)).view(B, T, -1, H, W)
        alphas = self.ModuleAlpha(fea, (H, W)).view(B, T, -1, H, W)  # B, T, ks^2, H, W
        betas = self.ModuleBeta(fea, (H, W)).view(B, T, -1, H, W)

        # * ====  RS differential time grid ===
        warp, GMs = [], []
        grid_y, _ = torch.meshgrid(
            torch.arange(0, H, device=frames[0].device, requires_grad=False),
            torch.arange(0, W, device=frames[0].device, requires_grad=False),
        )
        grid_sum = self.n_inputs - 1 + abs(tau - delta / H * grid_y)

        for i in range(self.n_inputs):
            weight = weights[:, i].contiguous()  # * (B, ks^2, H, W)
            alpha = alphas[:, i].contiguous()  # * (B, ks^2, H, W)
            beta = betas[:, i].contiguous()  # * (B, ks^2, H, W)
            occ = Occlusion[:, i : i + 1]  # * (B, 1, H, W)
            frame = F.interpolate(frames[i], size=weight.size()[-2:], mode="bilinear")

            # differential time grids weight
            grid = abs(tau + len(frames) // 2 - i - delta / H * grid_y)
            GMs.append((grid_sum - grid) * occ)
            warp.append(GMs[-1] * self.moduleAdaCoF(self.modulePad(frame), weight, alpha, beta, self.dilation))

        # warp = [(B3HW), (B3HW), (B3HW)....]
        frame = sum(warp) / sum(GMs)
        return frame


class FusionNet3D(nn.Module):
    def __init__(
        self,
        n_inputs=3,
        joinType="concat",
        ks=5,
        dilation=1,
        nf=[192, 128, 64, 32],
    ):
        super().__init__()
        ws = [(1, 8, 8), (1, 8, 8), (1, 8, 8), (1, 8, 8)]
        nh = [2, 4, 8, 16]
        nf_out = 64

        self.joinType = joinType
        self.n_inputs = n_inputs
        growth = 2 if joinType == "concat" else 1

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.encoder = SepSTS_Encoder(nf, n_inputs, window_size=ws, nh=nh)
        self.decoder = nn.Sequential(
            upSplit(nf[0], nf[1]),
            upSplit(nf[1] * growth, nf[2]),
            upSplit(nf[2] * growth, nf[3]),
        )

        def SmoothNet(inc, ouc):
            return torch.nn.Sequential(
                Conv_3d(inc, ouc, kernel_size=3, stride=1, padding=1, batchnorm=False),
                ResBlock(ouc, kernel_size=3),
            )

        self.smooth_ll = SmoothNet(nf[1] * growth, nf_out)
        self.smooth_l = SmoothNet(nf[2] * growth, nf_out)
        self.smooth = SmoothNet(nf[3] * growth, nf_out)

        self.predict_ll = RSAdaCof(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=True)
        self.predict_l = RSAdaCof(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=False)
        self.predict = RSAdaCof(n_inputs, nf_out, ks=ks, dilation=dilation, norm_weight=False)

    def forward(self, frames, corrfeilds, delta, tau):  # list(B,C,H,W)
        images = []
        for i in range(self.n_inputs):
            feild = corrfeilds[i]  # b, h, w, 2
            feild = rearrange(feild, "b h w c -> b c h w")
            images.append(torch.cat([frames[i], feild], dim=1))

        images = torch.stack(images, dim=2)  # B,C,T,H,W
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_

        x_0, x_1, x_2, x_3, x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4, x_3.size()))
        dx_3 = joinTensors(dx_3, x_3, type=self.joinType)

        dx_2 = self.lrelu(self.decoder[1](dx_3, x_2.size()))
        dx_2 = joinTensors(dx_2, x_2, type=self.joinType)

        dx_1 = self.lrelu(self.decoder[2](dx_2, x_1.size()))
        dx_1 = joinTensors(dx_1, x_1, type=self.joinType)

        fea3 = self.smooth_ll(dx_3)
        fea2 = self.smooth_l(dx_2)
        fea1 = self.smooth(dx_1)

        out_ll = self.predict_ll(fea3, frames, x_2.size()[-2:], delta, tau)

        out_l = self.predict_l(fea2, frames, x_1.size()[-2:], delta, tau)
        out_l = F.interpolate(out_ll, size=out_l.size()[-2:], mode="bilinear") + out_l

        out = self.predict(fea1, frames, x_0.size()[-2:], delta, tau)
        out = F.interpolate(out_l, size=out.size()[-2:], mode="bilinear") + out

        return out_ll, out_l, out


class QRST(nn.Module):
    def __init__(
        self,
        n_inputs=5,
        joinType="concat",
        ks=5,
        dilation=1,
        nf=[192, 128, 64, 32],
        config_file="/mnt/petrelfs/qudelin/.cache/mim/raft_8x2_100k_mixed_368x768.py",
        checkpoint_file="/mnt/petrelfs/qudelin/.cache/mim/raft_8x2_100k_mixed_368x768.pth",
    ):
        super().__init__()

        self.flowEstimator = init_model(config_file, checkpoint_file)
        for param in self.flowEstimator.named_parameters():
            param[1].requires_grad = False

        self.FusionNet3D = FusionNet3D(n_inputs=n_inputs - 2, joinType=joinType, ks=ks, dilation=dilation, nf=nf)
        self.n_inputs = n_inputs

    def forward(self, frames, gamma=1.0, tau=0, delta=1.0):  # frames: list[B,C,H,W]
        B, _, _, _ = frames[0].shape
        vals = [None for _ in range(B)]

        # * Quadratic Solver
        cvframes = [TensorToCV(f) for f in frames]

        # ? Quadratic Flows
        fpairs = []
        for i in range(self.n_inputs)[1:-1]:
            prevf = inference_model(self.flowEstimator, cvframes[i], cvframes[i - 1], valids=vals)
            postf = inference_model(self.flowEstimator, cvframes[i], cvframes[i + 1], valids=vals)

            # flow: B,H,W,2
            prevf = torch.stack([torch.from_numpy(flo["flow"]) for flo in prevf], dim=0).cuda()
            postf = torch.stack([torch.from_numpy(flo["flow"]) for flo in postf], dim=0).cuda()
            fpairs.append((prevf, postf))

        # * Quadratic Correction
        corrfeilds = []
        for i, fp in enumerate(fpairs):
            flow = quadratic_flow(fp[0], fp[1], gamma, tau + len(fpairs) // 2 - i)
            corrfeilds.append(flow)

        corrframes = []
        for i, feild in enumerate(corrfeilds):
            corrframes.append(feats_sampling(frames[i + 1], -feild))  # list(B,C,H,W)

        # * Fusion frames
        out = self.FusionNet3D(corrframes, corrfeilds, delta, tau)
        return out
