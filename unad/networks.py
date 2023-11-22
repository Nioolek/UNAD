import torch
import torch.nn as nn
from torch import Tensor


class REDCNNPlus(nn.Module):
    """RED-CNN+ Model with Distribution Regression Layer.

    Args:
        mid_ch (int): Number of channels in the middle layers.
        reg_max (int): The hyperparameter of Distribution Regression Layer.
            It means the number of Distribution.
        act (str): Activation function used in the model.
        pretrain (bool): Indicates whether we are in training or pretraining.
        y_0 (float): The minimum value of the predicted value.
        y_n (float): The maximum value of the predicted value.
        pretrain_out_ch (int): The number of channels of the output image in pre-training.
        norm_range_max (float): The normalization hyperparameter of the input image.
        norm_range_min (float): The normalization hyperparameter of the input image.
    """

    def __init__(self,
                 mid_ch: int = 96,
                 reg_max: int = 18,
                 act: str = 'leakyrelu',
                 pretrain: bool = False,
                 y_0: float = -160.,
                 y_n: float = 200.,
                 pretrain_out_ch: int = 6,
                 norm_range_max: float = 3072.,
                 norm_range_min: float = -1024.):
        super().__init__()
        self.reg_max = reg_max
        self.pretrain = pretrain
        self.y_0 = y_0
        self.y_n = y_n

        self.__defineNetworkStructure(mid_ch)

        self.__determineOutputLayer(mid_ch, y_0, y_n, pretrain_out_ch, norm_range_max, norm_range_min)

        self.__defineActivationFunc(act)

    def __determineOutputLayer(self, mid_ch, y_0, y_n, pretrain_out_ch, norm_range_max, norm_range_min):
        if self.pretrain:
            self.pretrain_out_ch = pretrain_out_ch
            self.output = nn.Conv2d(mid_ch, pretrain_out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            # Distribution Regression Layer
            self.output = nn.Conv2d(mid_ch, self.reg_max + 1, kernel_size=3, stride=1, padding=1, bias=True)
            # Bias is initialized to 1.0, which helps the model to accelerate convergence
            self.output.bias.data[:] = 1.0
            proj = torch.linspace(y_0, y_n, self.reg_max + 1, dtype=torch.float) / (norm_range_max - norm_range_min)
            self.register_buffer('proj', proj, persistent=False)
            self.hu_interval = (y_n - y_0) / self.reg_max

    def __defineActivationFunc(self, act):
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'silu':
            self.act = nn.SiLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise NotImplementedError

    def __defineNetworkStructure(self, mid_ch):
        self.conv1 = nn.Conv2d(1, mid_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)

        self.tconv5 = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=0)

    def forward(self, x: Tensor):
        # encoder
        residual_1 = x
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        residual_2 = out
        out = self.act(self.conv3(out))
        out = self.act(self.conv4(out))
        residual_3 = out
        out = self.act(self.conv5(out))

        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.act(out))
        out = self.tconv3(self.act(out))
        out += residual_2
        out = self.tconv4(self.act(out))
        out = self.tconv5(self.act(out))

        if self.pretrain:
            # In pre-training, the model does not use dfl and directly predicts before
            # and after images instead of predicting bias.
            out = self.output(out)
            return out
        else:
            # Distribution Regression Layer
            out = self.output(out)
            out_dist = out.permute(0, 2, 3, 1)
            out = out_dist.softmax(3).matmul(self.proj.view([-1, 1])).permute(0, 3, 1, 2)
            out += residual_1
            return out, out_dist


def __count_parameters(model: nn.Module):
    """Count the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = REDCNNPlus().cuda()

    para_num = __count_parameters(model)
    print('parameters num:', para_num)

    from thop import profile
    from ptflops import get_model_complexity_info

    input_64 = torch.randn(1, 1, 64, 64).cuda()
    input_512 = torch.randn(1, 1, 512, 512).cuda()

    flops, params = profile(model, inputs=(input_64,))

    print('FLOPs: ', flops)
    flops, params = profile(model, inputs=(input_512,))
    macs, params = get_model_complexity_info(model,
                                             (1, 64, 64),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
