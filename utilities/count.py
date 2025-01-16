import torch
#from torchstat import stat
#from models.net_factory import get_model
#model_name = 'u_net'
#model = get_model(model_name, class_num=4)


#print(stat(model, (3, 224, 224)))
#input_tensor = torch.randn(1, 3, 224, 224).cuda()


#stat(model, (3, 224, 224))


#output = model(input_tensor)
#print(f"Model output shape: {output.shape}")
group_outputs = None
for i in range(7):
    x = torch.randn(1, 8, 32, 32)
    if group_outputs is None:
        group_outputs = x
    else:
        group_outputs = torch.cat((group_outputs, x), dim=1)
print(group_outputs.shape)


