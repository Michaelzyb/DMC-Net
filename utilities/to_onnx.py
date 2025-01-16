import torch
import os
import sys
sys.path.append(os.path.abspath('..'))
from models.net_factory import get_model
import time
import onnxruntime
import numpy as np
import calflops
import os

def torch2onnx(model, out_path, dummy_input):
    model.eval()
    model_trace = torch.jit.trace(model, dummy_input)

    dynamic_axes_0 = {
        'input': {0: 'batch'},
        'output': {0: 'batch'},
    }
    # os.makedirs(out_path, exist_ok=True)
    torch.onnx.export(model_trace, dummy_input, out_path, dynamic_axes=dynamic_axes_0,
                      input_names=['input'], output_names=['output'], verbose=True)
    print(f'out_path: {out_path}_{model._get_name()}.onnx Finished!')
    input_shape = tuple([i for i in dummy_input.shape])
    flops, macs, paras = calflops.calculate_flops(model, input_shape=input_shape)
    print(f'flops: {flops}', f'paras: {paras}')
    return


def cac_fps(model_path, dummy_input):
    feature_session = onnxruntime.InferenceSession(model_path)
    num = 40
    dummy_input = {'input': dummy_input}
    for i in range(20):
        feature_session.run(None, dummy_input)
    start = time.time()
    for i in range(num):
        feature_session.run(None, dummy_input)
    fps = round(num / (time.time() - start), 3)
    print(f'FPS: {fps}, Inference time: {1000/fps}')
    return fps

if __name__ == '__main__':
    model_name = 'hdrnet'
    out_path = f'./{model_name}.onnx'
    size = (224, 224)
    dummy_input = torch.rand(1, 3, *size).cuda()
    model = get_model(model_name, class_num=4).cuda()
    torch2onnx(model, out_path, dummy_input)
    dummy_input = dummy_input.cpu().numpy().astype(np.float32)
    cac_fps(out_path, dummy_input)
    os.remove(out_path)
    # model_name = sys.argv[1]
    #model_name = 'u_net'
    #out_path = f'./{model_name}.onnx'
    #size = (512, 256)
    #dummy_input = torch.rand(1, 3, *size)
    #model = get_model(model_name, class_num=2)
    #torch2onnx(model, out_path, dummy_input)
    #dummy_input = dummy_input.numpy().astype(np.float32)
    #cac_fps(out_path, dummy_input)
    #os.remove(out_path)
