import torch.nn as nn


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
    super().__init__()

    self.block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, **kwargs)
        if down
        else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU()
        if use_act
        else nn.Identity()
    )

  def forward(self, x):
    x = self.block(x)
    return x


class ResBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()

    self.block = nn.Sequential(
        ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
        ConvBlock(in_channels, in_channels, use_act=False, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
    )

  def forward(self, x):
    x = x + self.block(x)
    return x

class Generator(nn.Module):
  def __init__(self, img_channels=3, num_features=64):
    super().__init__()

    self.initial = nn.Sequential(
        nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
        nn.ReLU()
    )

    self.down = nn.Sequential(
        ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
        ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1, padding_mode="reflect")
    )

    self.res = nn.Sequential(
        *[ResBlock(num_features*4) for i in range(9)]
    )

    self.up = nn.Sequential(
        ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
        ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
    )

    self.final = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

  def forward(self, x):
    x = self.initial(x)
    x = self.down(x)
    x = self.res(x)
    x = self.up(x)
    x = self.final(x)
    return x
  