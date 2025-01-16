from models.total_supvised import *
from models.total_supvised.hdrnet import *
from models.total_supvised.u_net import *



def get_model(net_type="uaps", in_chns=3, class_num=4):
    # total supervised model
    if net_type == 'hdrnet':
        net = HDR_Net(in_chns, class_num)
    elif net_type == 'u_net':
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == 'edrnet':
        net = EDRNet(in_chns, class_num)
    elif net_type == 'bisenet':
        net = BiSeNet(class_num)
    elif net_type == 'deeplabv3':
        net = DeepLabV3(in_chns, class_num)
    elif net_type == 'seg_net':
        net = SegNet(in_chns, class_num)
    elif net_type == 'swin_unet':
        net = SwinTransformerSys(in_chans=in_chns, num_classes=class_num, img_size=224)
    elif net_type == 'seg_former':
        net = segformer_b0(progress=False, num_classes=class_num)
    elif net_type == 'pga_net':
        net = PGANet(in_chns, class_num)
    elif net_type == 'topformer':
        net = TopFormer(class_num)
    else:
        AttributeError('Please input correct model name!')

    return net
