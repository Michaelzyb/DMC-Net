import cv2
import numpy as np
import os
import sys
sys.path.append('..')
from data.dataloaders import data_config, SurfaceDefectDataset
import albumentations as A
from PIL import Image
import torch
from torchvision import transforms as T
from models.net_factory import get_model


# This script is used to save the specified image in the log and record the metrics.
# Model result paths to be collected
benchmark = 'KolektorSDD2'
num_classes = 2
model_paths = {
    'u_net': '../logs/KolektorSDD2/u_net_0401_214616/u_net_KolektorSDD2_dice_0.6410854458808899.pth',
    # 'seg_net': '../logs/KolektorSDD2/seg_net'
    'a_net': '../logs/KolektorSDD2/a_net_0402_073621/a_net_KolektorSDD2_dice_0.6973711848258972.pth',
    'deeplabv3': '../logs/KolektorSDD2/deeplabv3_0402_031314/deeplabv3_KolektorSDD2_dice_0.7372789978981018.pth',
    'edrnet': '../logs/KolektorSDD2/edrnet_0414_112749/edrnet_KolektorSDD2_dice_0.6679368751287379.pth',
    'enet': '../logs/KolektorSDD2/enet_0402_183700/enet_KolektorSDD2_dice_0.749200165271759.pth'
}

# Specify file name
img_root = ''
label_root = ''
img_paths = ['']
label_paths = ['']
img_paths = [os.path.join(img_root, i) for i in img_paths]
label_paths = [os.path.join(label_root, i) for i in label_paths]

# output path
output_path = ''


def cac_miou(label, predict, num_classes):
    miou = 0
    for cls in range(1, num_classes):
        intersect = np.logical_and(label==cls, predict==cls).sum()
        union = np.logical_or(label==cls, predict==cls).sum()
        iou = intersect / (union + 1e-7)
        miou += iou.item()
    return miou / (num_classes - 1)


def mapping_color(img, num_classes):
    color_mapping = {
        # 0: [192, 192, 192],  # Class 0 - Light gray
        1: [120, 0, 0],  # Class 1 - Red
        2: [0, 120, 0],  # Class 2 - Green
        3: [120, 120, 0],  # Class 3 - Yellow
        4: [120, 0, 120],  # Class 4 - Magenta
        5: [0, 120, 120]  # Class 5 - Cyan
    }
    zero_mask = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
    for cls in range(1, num_classes):
        zero_mask[img == cls] = color_mapping[cls]
    return cv2.cvtColor(zero_mask, cv2.COLOR_BGR2RGB)


def load_imgs_labels(img_paths, label_paths, benchmark):
    imgs_batch = []
    labels_batch = []
    origin_images = []
    origin_labels = []
    for img_path, label_path in zip(img_paths, label_paths):
        img = cv2.imread(img_path)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if np.max(mask) > 128:
            mask = np.where(mask>128, 1, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        size = data_config[benchmark]['size']
        mean, std = data_config[benchmark]['mean'], data_config[benchmark]['std']
        t_val = A.Compose([A.Resize(size[0], size[1], interpolation=cv2.INTER_LINEAR)])
        aug = t_val(image=img, mask=mask)
        origin_image = Image.fromarray(aug['image'])
        origin_mask = aug['mask']
        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        img = t(origin_image)
        mask = torch.from_numpy(origin_mask)
        if mask.max() <= 1:
            mask = torch.where(mask > 0.5, 1., 0)
        mask = mask.long()

        imgs_batch.append(img)
        labels_batch.append(mask)
        origin_image = np.array(origin_image).astype(np.uint8)
        origin_images.append(origin_image)
        origin_labels.append(origin_mask.astype(np.uint8))

    imgs = torch.stack(imgs_batch)
    labels = torch.stack(labels_batch)
    return imgs, labels, origin_images, origin_labels


device = torch.device('cuda')
imgs, labels, origin_images, origin_labels = load_imgs_labels(img_paths, label_paths, benchmark)
imgs, labels = imgs.to(device), labels.to(device)


for model_name, model_weight in model_paths.items():
    model = get_model(model_name, class_num=num_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_weight))
    model.eval()
    predicts = model(imgs)
    if isinstance(predicts, (list, tuple)):
        predicts = predicts[0]
    predicts = torch.argmax(predicts, dim=1).cpu().numpy()
    for i, predict in enumerate(predicts):
        miou = cac_miou(origin_labels[i], predict, num_classes)
        img_name = img_paths[i].split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(output_path, f'{img_name}_{model_name}_{round(miou * 100, 1)}.png'), mapping_color(predict, num_classes))
        cv2.imwrite(os.path.join(output_path, f'{img_name}.png'), origin_images[i])
        cv2.imwrite(os.path.join(output_path, f'{img_name}_GT.png'), mapping_color(origin_labels[i], num_classes))