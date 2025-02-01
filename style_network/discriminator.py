import torch
import torch.nn as nn


class Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride, kernel_size=4) :
      super().__init__()

      self.conv_block = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=True, padding=1, padding_mode="reflect"),
          nn.InstanceNorm2d(out_channels),
          nn.LeakyReLU(0.2)
      )

  def forward(self, x):
    return self.conv_block(x)


class Discriminator(nn.Module):
  def __init__(self, in_channels=3, out_channels=[64, 128, 256, 512]):
    super().__init__()
    self.out_channels = out_channels

    self.initial = nn.Sequential(
          nn.Conv2d(in_channels, self.out_channels[0], kernel_size=4, stride=2, bias=True, padding=1, padding_mode="reflect"),
          nn.LeakyReLU(0.2)
      )

    self.final = nn.Conv2d(out_channels[-1], out_channels=1, kernel_size=4, stride=1, bias=True, padding=1, padding_mode="reflect")

  def forward(self, x):
    x = self.initial(x)
    in_channels = self.out_channels[0]

    for features in self.out_channels[1:]:
      block = Block(in_channels, features, stride = 1 if features == self.out_channels[-1] else 2).to(device)
      x = block(x)
      in_channels = features

    x = self.final(x)

    return torch.sigmoid(x)
