import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange




class LF_Blur(object):
    def __init__(self, kernel_size=21, blur_type='iso_gaussian',
                 sig_min=0.2, sig_max=4.0,
                 lambda_min=0.2, lambda_max=4.0,
                 sig=None, lambda_1=None, lambda_2=None,
                 theta=None):
        '''
        # sig, sig_min and sig_max are used for Isotropic Gaussian Blurs
        # the width of the blur kernel (i.e., sigma) is preferentially set to sig
        # if sig is None, sigma is randomly selected from [sig_min, sig_max]

        # lambda_min, lambda_max, lambda_1, lambda_2 and theta are used for Anisotropic Gaussian Blurs
        # if lambda_1 (eigenvalues of the covariance), lambda_2 and theta (angle value) are None,
        # lambda_1 and lambda_2 are randomly selected from [lambda_min, lambda_max]
        # theta is randomly selected from [0, pi]
        '''
        self.kernel_size = kernel_size
        self.blur_type = blur_type

        if blur_type == 'iso_gaussian':
            if sig is not None:
                sig_min, sig_max = sig, sig
            self.gen_kernel = Isotropic_Gaussian_Kernel(kernel_size, [sig_min, sig_max])

        elif blur_type == 'aniso_gaussian':
            if lambda_1 is not None and lambda_2 is not None and theta is not None:
                lambda_1_min, lambda_1_max = lambda_1, lambda_1
                lambda_2_min, lambda_2_max = lambda_2, lambda_2
                theta_min, theta_max = theta, theta
            else:
                lambda_1_min, lambda_1_max = lambda_min, lambda_max
                lambda_2_min, lambda_2_max = lambda_min, lambda_max
                theta_min, theta_max = 0, 180
            self.gen_kernel = Anisotropic_Gaussian_Kernel(kernel_size, [lambda_1_min, lambda_1_max],
                              [lambda_2_min, lambda_2_max], [theta_min, theta_max])

        ###
        self.blur_by_kernel = Blur_by_Kernel(kernel_size=kernel_size)


    def __call__(self, LF):
        [b, c, u, v, h, w] = LF.size()
        # 在B维度kernel是不相同的， 在C维度kernel相同
        LF = rearrange(LF, 'b c u v h w -> b (c u v) h w')
        [kernels, sigmas] = self.gen_kernel(LF.size(0))
        kernels = kernels.to(LF.device)

        LF_blured = self.blur_by_kernel(LF, kernels)

        # reshape
        LF_blured = rearrange(LF_blured, 'b (c u v) h w -> b c u v h w', u=u, v=v)

        return [LF_blured, kernels, sigmas]


class Blur_by_Kernel(nn.Module):
    def __init__(self, kernel_size=21):
        super(Blur_by_Kernel, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.padding = nn.ReflectionPad2d(kernel_size//2)
        else:
            self.padding = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))

    def forward(self, input, kernel):
        [B, C, H, W] = input.size()
        input_pad = self.padding(input)
        output = F.conv2d(input_pad.transpose(0, 1), kernel.unsqueeze(1), groups=B)
        output = output.transpose(0, 1)
        return output


class Isotropic_Gaussian_Kernel(object):
    def __init__(self, kernel_size, sig_range):
        self.kernel_size = kernel_size
        self.sig_min = sig_range[0]
        self.sig_max = sig_range[1]
        ax = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        xx = ax.repeat(self.kernel_size).view(1, self.kernel_size, self.kernel_size)
        yy = ax.repeat_interleave(self.kernel_size).view(1, self.kernel_size, self.kernel_size)
        self.xx_yy = -(xx ** 2 + yy ** 2)

    def __call__(self, batch, sig=None):
        # the width of the blur kernel (i.e., sigma) is randomly selected from [sig_min, sig_max]
        sig_min, sig_max = self.sig_min, self.sig_max
        sigma = torch.rand(batch) * (sig_max - sig_min) + sig_min + 1e-6

        kernel = torch.exp(self.xx_yy / (2. * sigma.view(-1, 1, 1) ** 2))
        kernel = kernel / kernel.sum([1, 2], keepdim=True)
        return kernel, sigma


class Anisotropic_Gaussian_Kernel(object):
    def __init__(self, kernel_size, lambda_1_range, lambda_2_range, theta_range):
        self.kernel_size = kernel_size
        self.lambda_1_range = lambda_1_range
        self.lambda_2_range = lambda_2_range
        self.theta_range = theta_range
        ax = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        xx = ax.repeat(self.kernel_size).view(1, self.kernel_size, self.kernel_size)
        yy = ax.repeat_interleave(self.kernel_size).view(1, self.kernel_size, self.kernel_size)
        self.xy = torch.stack([xx, yy], -1).view(1, -1, 2)

    def __call__(self, batch):
        # lambda_1, lambda_2, theta are randomly selected from [lambda_1_min, lambda_1_max],
        # [lambda_2_min, lambda_2_max], [theta_min, theta_max], respectively.
        lambda_1_min, lambda_1_max = self.lambda_1_range
        lambda_2_min, lambda_2_max = self.lambda_2_range
        theta_min, theta_max = self.theta_range

        lambda_1 = torch.rand(batch) * (lambda_1_max - lambda_1_min) + lambda_1_min
        lambda_2 = torch.rand(batch) * (lambda_2_max - lambda_2_min) + lambda_2_min
        theta = (torch.rand(batch) * (theta_max - theta_min) + theta_min) / 180 * math.pi

        covar = self.cal_sigma(lambda_1, lambda_2, theta)
        inverse_sigma = torch.inverse(covar)

        xy = self.xy.expand(batch, -1, -1)
        kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, self.kernel_size, self.kernel_size)
        kernel = kernel / kernel.sum([1, 2], keepdim=True)
        return kernel

    @staticmethod
    def cal_sigma(sig_x, sig_y, radians):
        sig_x = sig_x.view(-1, 1, 1)
        sig_y = sig_y.view(-1, 1, 1)
        radians = radians.view(-1, 1, 1)

        D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
        U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                       torch.cat([radians.sin(), radians.cos()], 2)], 1)
        sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))
        return sigma


def LF_Bicubic(LF, scale=1/4):
    [b, c, u, v, h, w] = LF.size()
    LF = rearrange(LF, 'b c u v h w -> (b u v) c h w')
    LF_bic = Bicubic()(LF, scale)
    LF_bic = rearrange(LF_bic, '(b u v) c h w -> b c u v h w', u=u, v=v)
    return LF_bic


# implementation of matlab bicubic interpolation in pytorch
class Bicubic(object):
    def __init__(self):
        super(Bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32)
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32)

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0)

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]), indice0), torch.FloatTensor([in_size[0]])).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]), indice1), torch.FloatTensor([in_size[1]])).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def __call__(self, input, scale=1/4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0].to(input.device)
        weight1 = weight1[0].to(input.device)

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = torch.sum(out, dim=3)
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out


class LF_Noise(object):
    def __init__(self, noise, random=False):
        self.noise = 0 if (noise is None) else noise
        self.random = random

    def __call__(self, LF):
        [b, c, u, v, h, w] = LF.size()
        if self.random:
            noise_level = torch.rand(b, 1, 1, 1, 1, 1).to(LF.device)
        else:
            noise_level = torch.ones(b, 1, 1, 1, 1, 1).to(LF.device)
        noise_level = noise_level * self.noise
        noise = torch.randn_like(LF).mul_(noise_level / 255)
        LF.add_(noise)

        return [LF, noise_level.squeeze()]


def random_crop_SAI(LF, LF_blured, SAI_patch):
    [_, _, U, V, H, W] = LF.size()
    bdr = 0
    h_idx = np.random.randint(bdr, H - SAI_patch - bdr + 1)
    w_idx = np.random.randint(bdr, W - SAI_patch - bdr + 1)
    LF = LF[:, :, :, :, h_idx: h_idx + SAI_patch, w_idx: w_idx + SAI_patch]
    LF_blured = LF_blured[:, :, :, :, h_idx: h_idx + SAI_patch, w_idx: w_idx + SAI_patch]
    return LF, LF_blured

