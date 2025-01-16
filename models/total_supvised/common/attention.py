import torch
from torch import nn
from torch.nn import Softmax


class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PosAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        att1 = self.conv1(x)
        att2 = self.conv2(x)
        att = self.act(att1 + att2)
        att = x * att
        return x + att


# 这个ChannelAttention和SE相比，多了一个max_out部分，是CBAM里的写法
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class NonLocalAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.k = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.q = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.v = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels == out_channels:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.shape
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        # 长宽维度展平后转置
        k = torch.permute(k.view(k.shape[0], k.shape[1], k.shape[2]*k.shape[3]), (0, 2, 1))
        q = q.view(q.shape[0], q.shape[1], q.shape[2]*q.shape[3])
        v = v.view(v.shape[0], v.shape[1], v.shape[2]*v.shape[3])

        attention = torch.softmax(torch.matmul(q, k), dim=2)

        out = torch.matmul(attention, v)
        out = out.reshape(n, c, h, w)

        res = self.res(x)
        return out + res


class AxisSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(AxisSpatialAttention, self).__init__()
        self.ver_conv = nn.Conv1d(in_channels*2, 1, kernel_size=7, padding=3)
        self.hor_conv = nn.Conv1d(in_channels*2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.shape
        tmp = x
        ver_mean_map = torch.mean(x, dim=3)
        ver_max_map = torch.max(x, dim=3)[0]
        ver_att = self.ver_conv(torch.cat((ver_max_map, ver_mean_map), dim=1))
        ver_att = self.softmax(ver_att)

        hor_mean_map = torch.mean(x, dim=2)
        hor_max_map = torch.max(x, dim=2)[0]
        hor_att = self.hor_conv(torch.cat((hor_max_map, hor_mean_map), dim=1))
        hor_att = self.softmax(hor_att)

        return tmp + ver_att.view(b, 1, h, 1) * tmp + tmp * hor_att.view(b, 1, 1, w)


class RowAttention(nn.Module):

    def __init__(self, in_dim, q_k_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(RowAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # size = (b*h,w,c2)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h,c2,w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # size = (b*h, c1,w)

        # size = (b*h,w,w) [:,i,j] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有h的第 Wj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        row_attn = torch.bmm(Q, K)
        ########
        # 此时的 row_atten的[:,i,0:w] 表示Q的所有h的第 Wi行位置上所有通道值与 K的所有行的 所有列(0:w)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:w）逐个位置的值的乘积，得到行attn
        ########

        # 对row_attn进行softmax
        row_attn = self.softmax(row_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*h,c1,w) 这里先需要对row_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 row_attn的行的乘积，即求权重和
        out = torch.bmm(V, row_attn.permute(0, 2, 1))

        # size = (b,c1,h,2)
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)

        out = self.gamma * out + x

        return out


class ColAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channel of input img tensor
        q_k_dim: int
            channel of Q, K vector
        device : torch.device
        '''
        super(ColAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        '''
        Parameters
        ----------
        x : Tensor
            4-D , (batch, in_dims, height, width) -- (b,c1,h,w)
        '''

        ## c1 = in_dims; c2 = q_k_dim
        b, _, h, w = x.size()

        Q = self.query_conv(x)  # size = (b,c2, h,w)
        K = self.key_conv(x)  # size = (b, c2, h, w)
        V = self.value_conv(x)  # size = (b, c1,h,w)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)

        # size = (b*w,h,h) [:,i,j] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的第 Hj列位置上的所有通道值的乘积，
        # 即(1,c2) * (c2,1) = (1,1)
        col_attn = torch.bmm(Q, K)
        ########
        # 此时的 col_atten的[:,i,0:w] 表示Q的所有W的第 Hi行位置上所有通道值与 K的所有W的 所有列(0:h)的逐个位置上的所有通道值的乘积
        # 此操作即为 Q的某个（i,j）与 K的（i,0:h）逐个位置的值的乘积，得到列attn
        ########

        # 对row_attn进行softmax
        col_attn = self.softmax(col_attn)  # 对列进行softmax，即[k,i,0:w] ，某一行的所有列加起来等于1，

        # size = (b*w,c1,h) 这里先需要对col_atten进行 行列置换，使得某一列的所有行加起来等于1
        # [:,i,j]即为V的所有行的某个通道上，所有列的值 与 col_attn的行的乘积，即求权重和
        out = torch.bmm(V, col_attn.permute(0, 2, 1))

        # size = (b,c1,h,w)
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)

        out = self.gamma * out + x

        return out


class AxisSelfAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super(AxisSelfAttention, self).__init__()
        self.row = RowAttention(in_dim, q_k_dim)
        self.col = ColAttention(in_dim, q_k_dim)

    def forward(self, x):
        x = self.row(x) + self.col(x) + x
        return x


class GolLocAttention(nn.Module):
    def __init__(self, in_channel, q_k_dim):
        super(GolLocAttention, self).__init__()
        self.channel_att = SE(in_channel, 8)
        self.local_att = SpatialAttention()
        self.row = RowAttention(in_channel, q_k_dim)
        self.col = ColAttention(in_channel, q_k_dim)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.row(x) + self.col(x) + self.local_att(x) + x
        return x


if __name__ == '__main__':
    device = torch.device('cuda')
    # 实现轴向注意力中的 Row Attention
    x = torch.randn(4, 8, 16, 20).to(device)
    row_attn = AxisSelfAttention(in_dim=8, q_k_dim=4).to(device)
    print(row_attn(x).size())
