<<<<<<< HEAD

import torch.fft
from einops import rearrange

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
import math
from timm.layers import DropPath

import matplotlib.pyplot as plt

class MSSIFF(nn.Module):
    def __init__(self, channels=4, band=30, out=64):
        super().__init__()

        self.c = channels
        # 光谱
        self.spectral1 = nn.Sequential(
            nn.Conv3d(1, self.c, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(self.c),
            nn.GELU()
        )
        self.spectral2 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 5), padding=(0, 0, 2)),
            nn.BatchNorm3d(self.c),
            nn.GELU()
        )

        # 空间
        self.spatial1 = nn.Sequential(
            nn.Conv3d(1, self.c, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(self.c),
            nn.GELU()
        )
        self.spatial2 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(self.c),
            nn.GELU()
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=2*band * channels, out_channels=out, kernel_size=1),
            nn.BatchNorm2d(out),
            nn.GELU()
        )

    def forward(self, x):

        x11, x12 = self.spectral1(x).chunk(2, dim=1)
        x21, x22 = self.spatial1(x).chunk(2, dim=1)

        x1 = torch.cat((x11, x21), dim=1)
        x2 = torch.cat((x12, x22), dim=1)

        y1 = self.spectral2(x1)
        y2 = self.spatial2(x2)

        y = torch.cat([y1, y2], dim=1)
        y = rearrange(y, 'b n h w c -> b (n c) h w')
        e = self.conv11(y)

        return e

import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        """
        构造函数，初始化 BiasFree LayerNorm 层。
        参数：
        - normalized_shape: 规范化的维度，通常为输入特征的最后一个维度。
        """
        super(BiasFree_LayerNorm, self).__init__()

        # 确保 normalized_shape 是一个整数，并转换为元组形式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 确保 normalized_shape 只有一个维度
        assert len(normalized_shape) == 1

        # 权重初始化为1，进行标准化
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        前向传播函数，执行无偏置的层归一化。
        """
        # 计算标准差（方差的平方根），避免负数的方差
        sigma = x.var(-1, keepdim=True, unbiased=False)

        # 返回归一化的张量（去均值，除以标准差）
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        """
        构造函数，初始化带有偏置的 LayerNorm 层。
        参数：
        - normalized_shape: 规范化的维度，通常为输入特征的最后一个维度。
        """
        super(WithBias_LayerNorm, self).__init__()

        # 确保 normalized_shape 是一个整数，并转换为元组形式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 确保 normalized_shape 只有一个维度
        assert len(normalized_shape) == 1

        # 权重初始化为1，偏置初始化为0
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        前向传播函数，执行带偏置的层归一化。
        """
        # 计算均值和方差
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        # 返回归一化后的张量，包含偏置和权重
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        """
        构造函数，选择性地应用无偏或带偏的 LayerNorm。
        参数：
        - dim: 输入特征的维度。
        - LayerNorm_type: 指定使用 'BiasFree' 或 'WithBias' 类型的 LayerNorm。
        """
        super(LayerNorm, self).__init__()

        # 根据 LayerNorm_type 初始化不同的 LayerNorm 实现
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)  # 无偏
        else:
            self.body = WithBias_LayerNorm(dim)  # 带偏

    def forward(self, x):
        """
        前向传播函数，应用 LayerNorm 并将输出恢复为原始的 4D 形状。
        """
        # 获取输入的高度和宽度
        h, w = x.shape[-2:]

        # 将输入转换为 3D 张量，通过归一化后再转换回原始的 4D 形状
        return to_4d(self.body(to_3d(x)), h, w)



class SSIA(nn.Module):
    # 定义构造函数，初始化模型参数
    def __init__(self, dim, num_heads=4, bias=False):
        super(SSIA, self).__init__()  # 调用父类的构造函数
        self.num_heads = num_heads  # 设置头的数量
        self.norm1 = LayerNorm(dim, LayerNorm_type='BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type='BiasFree')
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 初始化超参数

        # 定义查询、键和值的卷积层
        self.qkv = nn.Conv3d(dim, dim * 5, kernel_size=(1, 1, 1), bias=bias)
        # 深度可分离卷积层，用于qkv
        self.qkv_dwconv = nn.Conv3d(dim * 5, dim * 5, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 5, bias=bias)

        # 输出投影层
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    # 前向传播函数
    def forward(self, x):
        x = self.norm1(x)
        b, c, h, w = x.shape  # 获取输入张量的尺寸 torch.Size([1, 64, 32, 32])

        x = x.unsqueeze(2)  # 在第2维度增加一个维度，以便进行3D卷积  torch.Size([1, 64, 1, 32, 32])
        qkv1 = self.qkv_dwconv(self.qkv(x))  # 对输入执行QKV卷积，并通过深度可分离卷积
        qkv = qkv1.squeeze(2)  # 移除之前添加的维度   torch.Size([1, 64, 32, 32])

        # ======================全局特征======================
        q1, k1, v, q2, k2 = qkv.chunk(5, dim=1)  # 将qkv分割成三个部分：查询、键和值

        q1 = rearrange(q1, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        # 对q和k进行归一化处理
        q1 = torch.nn.functional.normalize(q1, dim=2)
        q2 = torch.nn.functional.normalize(q2, dim=2)
        k1 = torch.nn.functional.normalize(k1, dim=2)
        k2 = torch.nn.functional.normalize(k2, dim=2)

        # 计算注意力得分
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v)

        attn2 = (q2.transpose(-2, -1) @ k2) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        out2 = (v @ attn2)

        # 恢复输出形状
        out3 = rearrange(out1, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out2, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out3 + out4
        out = out.unsqueeze(2)  # 添加维度
        out = self.project_out(out)  # 投影输出
        out = out.squeeze(2)  # 移除维度

        out = self.norm2(out)
        out = self.proj(out)

        return out  # 返回最终输出


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.attn = SSIA(dim, num_heads=8)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        return x


def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


class Mix(nn.Module):
    # 初始化方法，设置初始混合权重m
    def __init__(self, m=0):
        super(Mix, self).__init__()  # 调用父类的初始化方法
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)  # 创建可学习参数w

        self.w = w  # 将w作为实例变量保存
        self.mix_block = nn.Sigmoid()  # 定义Sigmoid激活函数层

    # 前向传播方法，输入两个特征图fea1和fea2
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)  # 计算混合因子
        # 根据混合因子融合fea1和fea2
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out  # 返回融合后的特征图


class ACAF(nn.Module):
    def __init__(self, dim):
        super(ACAF, self).__init__()

        # 1x1卷积层，用于最终的输出投影
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out3 = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out4 = nn.Conv2d(dim, dim, kernel_size=1)

        self.c1 = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(dim)

        self.conv1 = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.k // 2)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.k // 2)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.k // 2)

        self.mix1 = Mix()
        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数层

    def forward(self, x1, x2):
        b, c, h, w = x1.shape  # 获取输入的形状

        # 通过 1x1 卷积进行输出投影
        out1 = self.project_out1(x1)
        out2 = self.project_out2(x2)

        # 对投影后的输出进行重排列，调整为适合自注意力计算的形状
        k1 = rearrange(out1, 'b c h w -> b h (w c)')
        v1 = rearrange(out1, 'b c h w -> b h (w c)')
        k2 = rearrange(out2, 'b c h w -> b w (h c)')
        v2 = rearrange(out2, 'b c h w -> b w (h c)')

        q2 = rearrange(out1, 'b c h w -> b w (h c)')
        q1 = rearrange(out2, 'b c h w -> b h (w c)')

        # 对q和k进行归一化处理
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        # 计算注意力得分
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1)

        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2)

        # 恢复输出形状
        out3 = rearrange(out3, 'b h (w c) -> b c h w', h=h, w=w)
        out4 = rearrange(out4, 'b w (h c) -> b c h w', h=h, w=w)

        # 最终输出，通过 1x1 卷积和输入的残差连接
        e1 = self.project_out3(out3)
        e2 = self.project_out4(out4)

        e = torch.cat((e1, e2), 1)
        e = self.c1(e)

        t1_channel_avg_pool = self.avg_pool(e).squeeze(-1).transpose(1, 2)
        t1_channel_max_pool = self.max_pool(e).squeeze(-1).transpose(1, 2)
        t1_channel_avg_pool = self.conv1(t1_channel_avg_pool).unsqueeze(-1).transpose(1, 2)
        t1_channel_max_pool = self.conv2(t1_channel_max_pool).unsqueeze(-1).transpose(1, 2)

        a1 = self.mix1(t1_channel_avg_pool, t1_channel_max_pool)
        a1 = self.conv3(a1.squeeze(-1).transpose(1, 2))

        w1 = self.sigmoid(a1.unsqueeze(-1).transpose(1, 2))
        f = e * w1

        return e


class Net(nn.Module):
    def __init__(self, band=30, channels=4, out=64, num_classes=16):
        super().__init__()
        self.mss = MSSIFF(band=band, channels=channels, out=out)

        self.conv11 = nn.Sequential(
            nn.Conv2d(band, out, 1),
            nn.BatchNorm2d(out),
            nn.GELU()
        )
        self.block = Block(out)
        self.bnge = nn.Sequential(
            nn.BatchNorm2d(out),
            nn.GELU()
        )

        self.ff = ACAF(out)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(out, num_classes),
        )

    def forward(self, x):
        b, _, h, w, _ = x.shape

        x1 = self.mss(x)

        y = rearrange(x, 'b n h w c -> b n c h w')
        y = y.squeeze(dim=1)
        y1 = self.conv11(y)
        y1 = self.block(y1)
        y1 = self.bnge(y1)

        e = self.ff(x1, y1)

        end = self.avg(e)
        end = end.view(end.size(0), -1)
        end = self.head(end)

        return end


if __name__ == '__main__':
    model = Net(band=30, channels=4, out=48, num_classes=16)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    input = torch.randn(64, 1, 17, 17, 30).cuda()
    y = model(input)
    print(y.shape)
=======

import torch.fft
from einops import rearrange

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
import math
from timm.layers import DropPath

import matplotlib.pyplot as plt

class MSSIFF(nn.Module):
    def __init__(self, channels=4, band=30, out=64):
        super().__init__()

        self.c = channels
        # 光谱
        self.spectral1 = nn.Sequential(
            nn.Conv3d(1, self.c, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(self.c),
            nn.GELU()
        )
        self.spectral2 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(1, 1, 5), padding=(0, 0, 2)),
            nn.BatchNorm3d(self.c),
            nn.GELU()
        )

        # 空间
        self.spatial1 = nn.Sequential(
            nn.Conv3d(1, self.c, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(self.c),
            nn.GELU()
        )
        self.spatial2 = nn.Sequential(
            nn.Conv3d(self.c, self.c, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(self.c),
            nn.GELU()
        )

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=2*band * channels, out_channels=out, kernel_size=1),
            nn.BatchNorm2d(out),
            nn.GELU()
        )

    def forward(self, x):

        x11, x12 = self.spectral1(x).chunk(2, dim=1)
        x21, x22 = self.spatial1(x).chunk(2, dim=1)

        x1 = torch.cat((x11, x21), dim=1)
        x2 = torch.cat((x12, x22), dim=1)

        y1 = self.spectral2(x1)
        y2 = self.spatial2(x2)

        y = torch.cat([y1, y2], dim=1)
        y = rearrange(y, 'b n h w c -> b (n c) h w')
        e = self.conv11(y)

        return e

import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        """
        构造函数，初始化 BiasFree LayerNorm 层。
        参数：
        - normalized_shape: 规范化的维度，通常为输入特征的最后一个维度。
        """
        super(BiasFree_LayerNorm, self).__init__()

        # 确保 normalized_shape 是一个整数，并转换为元组形式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 确保 normalized_shape 只有一个维度
        assert len(normalized_shape) == 1

        # 权重初始化为1，进行标准化
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        前向传播函数，执行无偏置的层归一化。
        """
        # 计算标准差（方差的平方根），避免负数的方差
        sigma = x.var(-1, keepdim=True, unbiased=False)

        # 返回归一化的张量（去均值，除以标准差）
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        """
        构造函数，初始化带有偏置的 LayerNorm 层。
        参数：
        - normalized_shape: 规范化的维度，通常为输入特征的最后一个维度。
        """
        super(WithBias_LayerNorm, self).__init__()

        # 确保 normalized_shape 是一个整数，并转换为元组形式
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # 确保 normalized_shape 只有一个维度
        assert len(normalized_shape) == 1

        # 权重初始化为1，偏置初始化为0
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        """
        前向传播函数，执行带偏置的层归一化。
        """
        # 计算均值和方差
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)

        # 返回归一化后的张量，包含偏置和权重
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        """
        构造函数，选择性地应用无偏或带偏的 LayerNorm。
        参数：
        - dim: 输入特征的维度。
        - LayerNorm_type: 指定使用 'BiasFree' 或 'WithBias' 类型的 LayerNorm。
        """
        super(LayerNorm, self).__init__()

        # 根据 LayerNorm_type 初始化不同的 LayerNorm 实现
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)  # 无偏
        else:
            self.body = WithBias_LayerNorm(dim)  # 带偏

    def forward(self, x):
        """
        前向传播函数，应用 LayerNorm 并将输出恢复为原始的 4D 形状。
        """
        # 获取输入的高度和宽度
        h, w = x.shape[-2:]

        # 将输入转换为 3D 张量，通过归一化后再转换回原始的 4D 形状
        return to_4d(self.body(to_3d(x)), h, w)



class SSIA(nn.Module):
    # 定义构造函数，初始化模型参数
    def __init__(self, dim, num_heads=4, bias=False):
        super(SSIA, self).__init__()  # 调用父类的构造函数
        self.num_heads = num_heads  # 设置头的数量
        self.norm1 = LayerNorm(dim, LayerNorm_type='BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type='BiasFree')
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 初始化超参数

        # 定义查询、键和值的卷积层
        self.qkv = nn.Conv3d(dim, dim * 5, kernel_size=(1, 1, 1), bias=bias)
        # 深度可分离卷积层，用于qkv
        self.qkv_dwconv = nn.Conv3d(dim * 5, dim * 5, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 5, bias=bias)

        # 输出投影层
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    # 前向传播函数
    def forward(self, x):
        x = self.norm1(x)
        b, c, h, w = x.shape  # 获取输入张量的尺寸 torch.Size([1, 64, 32, 32])

        x = x.unsqueeze(2)  # 在第2维度增加一个维度，以便进行3D卷积  torch.Size([1, 64, 1, 32, 32])
        qkv1 = self.qkv_dwconv(self.qkv(x))  # 对输入执行QKV卷积，并通过深度可分离卷积
        qkv = qkv1.squeeze(2)  # 移除之前添加的维度   torch.Size([1, 64, 32, 32])

        # ======================全局特征======================
        q1, k1, v, q2, k2 = qkv.chunk(5, dim=1)  # 将qkv分割成三个部分：查询、键和值

        q1 = rearrange(q1, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        q2 = rearrange(q2, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k1 = rearrange(k1, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        # 对q和k进行归一化处理
        q1 = torch.nn.functional.normalize(q1, dim=2)
        q2 = torch.nn.functional.normalize(q2, dim=2)
        k1 = torch.nn.functional.normalize(k1, dim=2)
        k2 = torch.nn.functional.normalize(k2, dim=2)

        # 计算注意力得分
        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature
        attn1 = attn1.softmax(dim=-1)
        out1 = (attn1 @ v)

        attn2 = (q2.transpose(-2, -1) @ k2) * self.temperature
        attn2 = attn2.softmax(dim=-1)
        out2 = (v @ attn2)

        # 恢复输出形状
        out3 = rearrange(out1, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out2, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = out3 + out4
        out = out.unsqueeze(2)  # 添加维度
        out = self.project_out(out)  # 投影输出
        out = out.squeeze(2)  # 移除维度

        out = self.norm2(out)
        out = self.proj(out)

        return out  # 返回最终输出


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.attn = SSIA(dim, num_heads=8)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        return x


def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


class Mix(nn.Module):
    # 初始化方法，设置初始混合权重m
    def __init__(self, m=0):
        super(Mix, self).__init__()  # 调用父类的初始化方法
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)  # 创建可学习参数w

        self.w = w  # 将w作为实例变量保存
        self.mix_block = nn.Sigmoid()  # 定义Sigmoid激活函数层

    # 前向传播方法，输入两个特征图fea1和fea2
    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)  # 计算混合因子
        # 根据混合因子融合fea1和fea2
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out  # 返回融合后的特征图


class ACAF(nn.Module):
    def __init__(self, dim):
        super(ACAF, self).__init__()

        # 1x1卷积层，用于最终的输出投影
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out3 = nn.Conv2d(dim, dim, kernel_size=1)
        self.project_out4 = nn.Conv2d(dim, dim, kernel_size=1)

        self.c1 = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(dim)

        self.conv1 = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.k // 2)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.k // 2)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=self.k, padding=self.k // 2)

        self.mix1 = Mix()
        self.sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数层

    def forward(self, x1, x2):
        b, c, h, w = x1.shape  # 获取输入的形状

        # 通过 1x1 卷积进行输出投影
        out1 = self.project_out1(x1)
        out2 = self.project_out2(x2)

        # 对投影后的输出进行重排列，调整为适合自注意力计算的形状
        k1 = rearrange(out1, 'b c h w -> b h (w c)')
        v1 = rearrange(out1, 'b c h w -> b h (w c)')
        k2 = rearrange(out2, 'b c h w -> b w (h c)')
        v2 = rearrange(out2, 'b c h w -> b w (h c)')

        q2 = rearrange(out1, 'b c h w -> b w (h c)')
        q1 = rearrange(out2, 'b c h w -> b h (w c)')

        # 对q和k进行归一化处理
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)

        # 计算注意力得分
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1)

        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2)

        # 恢复输出形状
        out3 = rearrange(out3, 'b h (w c) -> b c h w', h=h, w=w)
        out4 = rearrange(out4, 'b w (h c) -> b c h w', h=h, w=w)

        # 最终输出，通过 1x1 卷积和输入的残差连接
        e1 = self.project_out3(out3)
        e2 = self.project_out4(out4)

        e = torch.cat((e1, e2), 1)
        e = self.c1(e)

        t1_channel_avg_pool = self.avg_pool(e).squeeze(-1).transpose(1, 2)
        t1_channel_max_pool = self.max_pool(e).squeeze(-1).transpose(1, 2)
        t1_channel_avg_pool = self.conv1(t1_channel_avg_pool).unsqueeze(-1).transpose(1, 2)
        t1_channel_max_pool = self.conv2(t1_channel_max_pool).unsqueeze(-1).transpose(1, 2)

        a1 = self.mix1(t1_channel_avg_pool, t1_channel_max_pool)
        a1 = self.conv3(a1.squeeze(-1).transpose(1, 2))

        w1 = self.sigmoid(a1.unsqueeze(-1).transpose(1, 2))
        f = e * w1

        return e


class Net(nn.Module):
    def __init__(self, band=30, channels=4, out=64, num_classes=16):
        super().__init__()
        self.mss = MSSIFF(band=band, channels=channels, out=out)

        self.conv11 = nn.Sequential(
            nn.Conv2d(band, out, 1),
            nn.BatchNorm2d(out),
            nn.GELU()
        )
        self.block = Block(out)
        self.bnge = nn.Sequential(
            nn.BatchNorm2d(out),
            nn.GELU()
        )

        self.ff = ACAF(out)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(out, num_classes),
        )

    def forward(self, x):
        b, _, h, w, _ = x.shape

        x1 = self.mss(x)

        y = rearrange(x, 'b n h w c -> b n c h w')
        y = y.squeeze(dim=1)
        y1 = self.conv11(y)
        y1 = self.block(y1)
        y1 = self.bnge(y1)

        e = self.ff(x1, y1)

        end = self.avg(e)
        end = end.view(end.size(0), -1)
        end = self.head(end)

        return end


if __name__ == '__main__':
    model = Net(band=30, channels=4, out=48, num_classes=16)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    input = torch.randn(64, 1, 17, 17, 30).cuda()
    y = model(input)
    print(y.shape)
>>>>>>> b1171db (first commit)
