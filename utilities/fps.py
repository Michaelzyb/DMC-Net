import os.path
import time
import torch
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath('..'))
from models.net_factory import get_model


def export_onnx(model, out_path, dummy_input):
    model.eval()
    model_trace = torch.jit.trace(model, dummy_input)

    dynamic_axes_0 = {
        'input': {0: 'batch'},
        'output': {0: 'batch'},
    }
    # os.makedirs(out_path, exist_ok=True)
    torch.onnx.export(model_trace, dummy_input, f'{out_path}_{model._get_name()}.onnx',
                      input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes_0, verbose=True)
    print(f'out_path: {out_path}_{model._get_name()}.onnx Finished!')
    return


def fastsurfacenet_onnx(model, out_path, dummy_input):
    model.eval()

    seg_model = model.seg
    model_trace = torch.jit.trace(model, dummy_input)

    dynamic_axes_0 = {
        'input': {0: 'batch'},
        'output': {0: 'batch'},
    }
    # os.makedirs(out_path, exist_ok=True)
    torch.onnx.export(model_trace, dummy_input, f'{out_path}_{model._get_name()}.onnx',
                      input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes_0, verbose=True)
    print(f'out_path: {out_path}_{model._get_name()}.onnx Finished!')
    return


def cac_fps(model, size=(224, 224), device='cuda'):
    model.to(device)
    with torch.no_grad():
        model.eval()
        sample = torch.rand(1, 3, *size)
        sample = sample.to(device)
        num = 100
        for i in range(80):
            _ = model(sample)

        start = time.time()
        for i in range(num):
            _ = model(sample)

    fps = round(num / (time.time() - start), 3)
    print(f'FPS: {fps}')
    return fps


def cac_fastsurface_fps(model, size=(512, 256), device='cpu', ratio=0.9):
    model.to(device)
    with torch.no_grad():
        model.eval()
        sample = torch.rand(1, 3, *size)
        sample = sample.to(device)
        num = 600
        for i in range(80):
            _ = model(sample)

        start = time.time()
        for i in range(num):
            e1, e2, e3 = model.feature(sample)
            _ = model.cls(e3)
            if i >= ratio*num:
                out = model.seg(e1, e2, e3)
                out = F.interpolate(out[0], size=size, mode='bilinear', align_corners=True)


    fps = round(num / (time.time() - start), 3)
    print(f'FPS: {fps}')
    return fps


#model_name = sys.argv[1]
model = get_model('hdrnet', class_num=4)
# model.mode = 'cls'
# s = torch.rand(1, 3, 224, 224)
size = (224, 224)
# export_onnx(model, '', s)
cac_fps(model, size, device='cuda')

# import numpy as np
# import onnxruntime
# feature_path = '../feature_cls.onnx'
# seg_path = '../seg_model.onnx'
# feature_session = onnxruntime.InferenceSession(feature_path)
# seg_session = onnxruntime.InferenceSession(seg_path)
#
# def onnx_fps(size):
#     input_dict = {'input': np.random.rand(1, 3, *size).astype(np.float32)}
#     start = time.time()
#     for i in range(500):
#         e1, e2, e3, score = feature_session.run(None, input_dict)
#         if i > 250:
#             seg_input_dict = {'e1': e1, 'e2': e2, 'e3': e3}
#             result = seg_session.run(None, seg_input_dict)
#     end = time.time()
#     fps = 500 / (end-start)
#     print(fps)
#
# onnx_fps((512, 256))