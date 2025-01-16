from models.net_factory import get_model
import torch
from thop import profile
def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


model_name = 'u_net'
#input_shape = (1, 3, 224, 224)
model = get_model(model_name, class_num=4).cuda()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params / 1e6}")

input_tensor = torch.randn(1, 3, 224, 224).cuda()

flops, params = profile(model, inputs=(input_tensor,))
flops = flops / 1e9
params = params / 1e6
print(f"Floating Point Operations: {flops}")
print(f"Number of parameters: {params}")