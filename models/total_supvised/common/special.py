import torch


# 一些特殊运算在这个文件
def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()

    # 如果通道数不是分组的整数被，则无法进行Channel Shuffle操作，直接返回x
    if chnls % groups:
        raise AttributeError('Please confirm channels can be exact division!')

    # 计算用于Channel Shuffle的一个group的的通道数
    chnls_per_group = chnls // groups

    # 执行channel shuffle操作，不要直接用view变成5个维度，导出的onnx会报错
    x = x.unsqueeze(1)
    x = x.view(bs, groups, chnls_per_group, h, w)  # 将通道那个维度拆分为 (g,n) 两个维度
    x = torch.transpose(x, 1, 2).contiguous()  # 将这两个维度转置变成 (n,g)
    x = x.view(bs, -1, h, w)  # 最后重新reshape成一个维度 g × n g\times ng×n

    return x