import os
import cv2
import numpy as np
import torch
from PIL import Image
import sys
sys.path.append(os.path.abspath('..'))
from models.net_factory import get_model
from torchvision import transforms
import albumentations as A
from data.dataloaders import data_config


benchmark = 'neuseg'
target_category = [2]  # tabby, tabby cat
model_name = 'hdrnet'
mode = 'segmentation'
model = get_model(model_name, class_num=4)
weight_path = ''
target_layers = []
if model_name == 'deeplabv3':
    target_layers = [model.encoder.resnet[6][0].conv3]
    weight_path = ''
elif model_name == 'hdrnet':
    target_layers = [model.conv1x1]
    weight_path = ''
elif model_name == 'u_net':
    target_layers = [model.decoder.up1.conv1x1]
    weight_path = ''
model.load_state_dict(torch.load(weight_path))
model.eval()


img_path = ''
batch_size = 1

size = data_config[benchmark]['size']
mean = data_config[benchmark]['mean']
std = data_config[benchmark]['std']
# val transform template
t_val = A.Compose([A.Resize(size[0], size[1], interpolation=cv2.INTER_LINEAR)])
T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

img = cv2.imread(img_path)
aug = t_val(image=img)
x = Image.fromarray(aug['image'])
x = T(x).unsqueeze(0)
x.require_grad = True


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 criteon,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.criteon = criteon
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    def get_loss(self, output, target_category, mode='classification'):
        loss = 0
        # mode == 'segmentation':
        if isinstance(output, (tuple, list)):
                output = output[0]
        if isinstance(target_category, (tuple, list)):
            one_hot = torch.zeros([output.shape[0], output.shape[1], 1, 1])
            for tar_cls in target_category:
                one_hot[0, tar_cls] = 1
            one_hot.requires_grad = True
            loss = output * one_hot
        else:
            loss = output * target_category
        return loss.sum()

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None, mode='classification'):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        # else:
        #     assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category, mode)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap*0.005 + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img


criteon = torch.nn.CrossEntropyLoss()
cam = GradCAM(model=model, target_layers=target_layers, criteon=criteon, use_cuda=False)
# target_category = 254  # pug, pug-dog

grayscale_cam = cam(input_tensor=x, target_category=target_category, mode=mode)

grayscale_cam = grayscale_cam[0, :]
for i in range(3):
    if len(std) == 3:
        x[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    else:
        x[:, i, :, :] = x[:, i, :, :] * std[0] + mean[0]

x = torch.permute(x.squeeze(), (1, 2, 0)).numpy()

# grayscale_cam = grayscale_cam.max() - grayscale_cam
visualization = show_cam_on_image(x.astype(dtype=np.float32) / 255.,
                                  grayscale_cam,
                                  use_rgb=True)
grayscale_cam = Image.fromarray(visualization)
origin_img = Image.fromarray(x.astype(dtype=np.uint8))
grayscale_cam.save(f'./gradcam_{model_name}_{img_path.split("/")[-1]}')
# grayscale_cam *= 255
# grayscale_cam.save(f'../data_show/cam/{benchmark}_{index}.jpg')
