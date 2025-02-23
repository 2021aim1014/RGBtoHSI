
import torch
from torch import nn
from torch.nn import functional as F

# Attention Weighted Channel Aggregation (AWCA) module
class AWCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AWCA, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 1, bias=False)  # Convolution to generate attention weights
        self.softmax = nn.Softmax(dim=2)  # Softmax for normalization
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        mask = self.conv(x).view(b, 1, h * w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(x.view(b, c, h * w).unsqueeze(1), mask).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Non-Local Attention Block
class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(NONLocalBlock2D, self).__init__()
        self.inter_channels = in_channels // reduction
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.W = nn.Conv2d(self.inter_channels, in_channels, 1, bias=False)
        nn.init.constant_(self.W.weight, 0)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        f = torch.matmul(theta_x, theta_x.transpose(1, 2))
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, H, W)
        return self.W(y) + x

# Position-aware Non-Local Block
class PSNL(nn.Module):
    def __init__(self, channels):
        super(PSNL, self).__init__()
        self.non_local = NONLocalBlock2D(channels)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        H1, W1 = H // 2, W // 2
        output = torch.zeros_like(x)
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            output[:, :, i * H1:(i + 1) * H1, j * W1:(j + 1) * W1] = self.non_local(x[:, :, i * H1:(i + 1) * H1, j * W1:(j + 1) * W1])
        return output

# 3x3 Convolution with Reflection Padding
class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        self.pad = nn.ReflectionPad2d(dilation * (kernel_size - 1) // 2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        return self.conv(self.pad(x))

# Deep Residual Attention Block
class DRAB(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim):
        super(DRAB, self).__init__()
        self.conv1 = Conv3x3(in_dim, in_dim, 3, 1)
        self.conv2 = Conv3x3(in_dim, in_dim, 3, 1)
        self.up_conv = Conv3x3(in_dim, res_dim, 5, 1)
        self.se = AWCA(res_dim)
        self.down_conv = Conv3x3(res_dim, out_dim, 3, 1)
        self.relu = nn.PReLU()

    def forward(self, x, res):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x) + identity
        x = self.relu(self.up_conv(x) + res)
        res = x
        x = self.se(x)
        return self.relu(self.down_conv(x) + identity), res

# Attention Weighted Aggregation Network (AWAN)
class AWAN(nn.Module):
    def __init__(self, inplanes=3, planes=31, channels=96, n_DRBs=8):
        super(AWAN, self).__init__()
        self.input_conv = Conv3x3(inplanes, channels, 3, 1)
        self.backbone = nn.ModuleList([DRAB(channels, channels, channels) for _ in range(n_DRBs)])
        self.output_conv = Conv3x3(channels, planes, 3, 1)
        self.tail_nonlocal = PSNL(planes)
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.relu(self.input_conv(x))
        residual = out
        res = out
        for block in self.backbone:
            out, res = block(out, res)
        out = self.output_conv(out + residual)
        return self.tail_nonlocal(out)

# Unit tests
if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 32, 32)
    assert AWCA(64)(input_tensor).shape == input_tensor.shape, "AWCA test failed"
    
    assert NONLocalBlock2D(64)(input_tensor).shape == input_tensor.shape, "NONLocalBlock2D test failed"
    
    assert PSNL(64)(input_tensor).shape == input_tensor.shape, "PSNL test failed"
    
    assert Conv3x3(64, 128, 3, 1)(input_tensor).shape == (1, 128, 32, 32), "Conv3x3 test failed"
    
    input_tensor = torch.randn(1, 96, 32, 32)
    output, res = DRAB(96, 96, 96)(input_tensor, input_tensor)
    assert output.shape == input_tensor.shape, "DRAB test failed"
    assert res.shape == input_tensor.shape, "DRAB residual test failed"
    
    b, c, h, w = 1, 3, 128, 128
    b, output_c, h, w =  1, 31, 128, 128
    input_image = torch.randn(b, c, h, w)
    assert AWAN()(input_image).shape == (b, output_c, h, w), "AWAN test failed"
    
    print("All tests passed!")
