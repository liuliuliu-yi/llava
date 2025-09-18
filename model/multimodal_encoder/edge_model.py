import torch
import torch.nn as nn
from xresnet1d_101 import ResBlock, ConvLayer, NormType, init_cnn

class GradualResBlock(nn.Module):
    """
    渐进升维+残差结构Block
    """
    def __init__(self, in_c, out_c, stride=2, kernel_size=3, norm_type=NormType.Batch, act_cls=nn.ReLU):
        super().__init__()
        self.block = ResBlock(
            expansion=1, ni=in_c, nf=out_c, stride=stride, kernel_size=kernel_size,
            norm_type=norm_type, act_cls=act_cls, ndim=1
        )
    def forward(self, x):
        return self.block(x)

class EdgeXResNet1d(nn.Module):
    def __init__(
        self,
        input_channels=12,
        num_classes=105,
        high_layers_channels=(512, 768, 1024, 1536),  # 3层渐进升维
        high_layers_kernel_size=3,
        mid_c=256,
        pool_type='avg'
    ):
        super().__init__()
        # === xresnet1d_101 前m层 ===
        self.stem = nn.Sequential(
            ConvLayer(input_channels, 32, ks=5, stride=2, ndim=1),
            ConvLayer(32, 32, ks=5, stride=1, ndim=1),
            ConvLayer(32, 64, ks=5, stride=1, ndim=1)
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block1 = nn.Sequential(
            ResBlock(4, 16, 64, stride=1, kernel_size=5, ndim=1),
            ResBlock(4, 64, 64, stride=1, kernel_size=5, ndim=1),
            ResBlock(4, 64, 64, stride=1, kernel_size=5, ndim=1)
        )
        self.block2 = nn.Sequential(
            ResBlock(4, 64, 128, stride=2, kernel_size=5, ndim=1),
            ResBlock(4, 128, 128, stride=1, kernel_size=5, ndim=1),
            ResBlock(4, 128, 128, stride=1, kernel_size=5, ndim=1),
            ResBlock(4, 128, 128, stride=1, kernel_size=5, ndim=1)
        )
        # === 高层渐进升维+残差结构 ===
        # block2输出通道应为512
        in_chs = high_layers_channels[:-1]
        out_chs = high_layers_channels[1:]
        self.high_blocks = nn.Sequential(
            *[GradualResBlock(in_c, out_c, stride=2, kernel_size=high_layers_kernel_size)
              for in_c, out_c in zip(in_chs, out_chs)]
        )
        # === 分类头 ===
        if pool_type == 'concat':
            self.head_pool = nn.AdaptiveConcatPool1d(1)
            fc_in = high_layers_channels[-1] * 2
        else:
            self.head_pool = nn.AdaptiveAvgPool1d(1)
            fc_in = high_layers_channels[-1]
        self.head = nn.Sequential(
            self.head_pool,
            nn.Flatten(),
            nn.Linear(fc_in, mid_c),
            nn.ReLU(inplace=True),
            nn.Linear(mid_c, num_classes)
        )

        init_cnn(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        # x.shape: [B, 512, L]
        x = self.high_blocks(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = EdgeXResNet1d(
        input_channels=12, num_classes=105,
        high_layers_channels=(512, 768, 1024, 1536),  # 3层升维
        pool_type='avg'
    )
    x = torch.randn(8, 12, 5000)  # batch=8, 12导联, 5000采样点
    out = model(x)
    print(out.shape)  # [8, 105]