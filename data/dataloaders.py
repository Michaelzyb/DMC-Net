import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import cv2
import albumentations as A
import os

data_config = {
            'KolektorSDD2': {'size': [512, 256], 'mean': [0.1753, 0.1708, 0.1770], 'std': [0.0436, 0.0377, 0.0370]},
            'MT': {'size': [256, 256], 'mean': [0.432], 'std': [0.1814]},
            'carpet': {'size': [256, 256], 'mean': [0.3688, 0.3524, 0.3563], 'std': [0.1410, 0.1362, 0.1221]},
            'neuseg': {'size': [224, 224], 'mean': [0.4326], 'std': [0.2039]},
            'hazelnut': {'size': [256, 256], 'mean': [0.2434, 0.1816, 0.1768], 'std': [0.1653, 0.0767, 0.0456]},
            'CrackForest': {'size': [320, 480], 'mean': [0.5448, 0.5248, 0.5085], 'std': [0.0734, 0.0714, 0.0806]},
            'CDD': {'size': [384, 544], 'mean': [0.5978, 0.5846, 0.5651]},
            'DAGM': {'size': [256, 256], 'mean': [[0.2744], [0.3961], [0.5166], [0.6927], [0.5000], [0.3808], [0.7609], [0.4294], [0.4965], [0.6175]], 'std': []},
            'RSDD1': {'size': [800, 160], 'mean': [0.3903], 'std': [0.2797]},
            'severstal': {'size': [256, 1600], 'mean': [0.3453], 'std': [0.1961]}
        }


class SurfaceDefectDataset(Dataset):
    def __init__(self, img_path, mask_path, mean, std, benchmark, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.benchmark = benchmark
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        if img is None:
            print(self.img_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if os.path.exists(self.mask_path[idx]):
            mask = cv2.imread(self.mask_path[idx], cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        if np.max(mask) > 128:
            mask = np.where(mask>128, 1, 0)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask)
        if mask.max() > 20:
            mask = torch.where(mask > 127, 1., 0)
        if mask.max() <= 1:
            mask = torch.where(mask > 0.5, 1., 0)
        if self.benchmark == 'MT':
            mask = self.get_class(mask, self.mask_path[idx])
        mask = mask.long()
        return img, mask

    def get_class(self, mask, mask_name):
        if self.benchmark == 'MT':
            if 'Blowhole' in mask_name:
                mask[mask == 1] = 1
            elif 'Crack' in mask_name:
                mask[mask == 1] = 2
            elif 'Fray' in mask_name:
                mask[mask == 1] = 3
        return mask


def get_loaders(benchmark, root_path, batch_size, mode='semi-sup', unlabeled_ratio=0.4):
    if 'DAGM' not in benchmark:
        size = data_config[benchmark]['size']
        mean, std = data_config[benchmark]['mean'], data_config[benchmark]['std']
    if benchmark == 'KolektorSDD':
        imgs_masks = get_kolektorsdd(root_path, unlabeled_ratio, mode)
    elif benchmark == 'KolektorSDD2':
        imgs_masks = get_kolektorsdd2(root_path, unlabeled_ratio, mode)
    elif benchmark == 'MT':
        imgs_masks = get_magnetic(root_path, unlabeled_ratio, mode)
    elif benchmark == 'carpet':
        imgs_masks = get_carpet(root_path, unlabeled_ratio, mode)
    elif benchmark == 'neuseg':
        imgs_masks = get_neuseg(root_path, unlabeled_ratio, mode)
    elif benchmark == 'hazelnut':
        imgs_masks = get_carpet(root_path, unlabeled_ratio, mode)
    elif benchmark == 'CrackForest':
        imgs_masks = get_cfd(root_path, unlabeled_ratio, mode)
    elif benchmark == 'CDD':
        imgs_masks = get_cdd(root_path, unlabeled_ratio, mode)
    elif benchmark == 'RSDD1':
        imgs_masks = get_rsdd(root_path, unlabeled_ratio, mode)
    elif benchmark == 'severstal':
        imgs_masks = get_severstal(root_path, unlabeled_ratio, mode)
    elif 'DAGM' in benchmark:
        class_id = int(benchmark.split('DAGM')[1])
        benchmark = 'DAGM'
        root_path = os.path.join(root_path, 'Class' + str(class_id))
        imgs_masks = get_dagm(root_path, unlabeled_ratio, mode)
        size = data_config[benchmark]['size']
        mean, std = data_config[benchmark]['mean'][class_id-1], data_config[benchmark]['std'][class_id-1]
    else:
        raise print('Please input correct benchmark!')

    print('Train Size   : ', len(imgs_masks['imgs'][0]))
    print('Val Size     : ', len(imgs_masks['imgs'][1]))
    print('Test Size    : ', len(imgs_masks['imgs'][2]))
    t_train = A.Compose([A.Resize(size[0], size[1], interpolation=cv2.INTER_LINEAR),
                         A.HorizontalFlip(p=0.3),
                         A.VerticalFlip(p=0.3),
                         # A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                         A.Blur(p=0.3),
                         A.GaussNoise(p=0.3),
                         ])
    t_val = A.Compose([A.Resize(size[0], size[1], interpolation=cv2.INTER_LINEAR)])

    # datasets
    train_set = SurfaceDefectDataset(imgs_masks['imgs'][0], imgs_masks['masks'][0], mean, std, benchmark, t_train)
    val_set = SurfaceDefectDataset(imgs_masks['imgs'][1], imgs_masks['masks'][1], mean, std, benchmark, t_val)
    test_set = SurfaceDefectDataset(imgs_masks['imgs'][2], imgs_masks['masks'][2], mean, std, benchmark, t_val)

    # dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=batch_size, pin_memory=False, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=batch_size, pin_memory=False, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=batch_size, pin_memory=False, shuffle=False)


    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


def get_kolektorsdd(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    train_imgs = []
    test_imgs = []
    test_masks = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if 'Train' in file_path:
                if '.jpg' in file_path:
                    train_imgs.append(file_path)
            if 'Test' in file_path:
                if '.jpg' in file_path:
                    test_imgs.append(file_path)
                    test_masks.append(file_path.replace('.jpg', '_label.bmp'))

    train_imgs, val_imgs = train_test_split(np.array(train_imgs), test_size=0.20, random_state=69)
    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            ret.append(file.replace('.jpg', '_label.bmp'))
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)


    if mode == 'semi-sup':#We do not include semi-supervision in this program, and we will participate in semi-supervised research in the future
        return {'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
                'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]}
    else:
        return {'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]}


def get_kolektorsdd2(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    train_imgs = []
    test_imgs = []
    test_masks = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if 'train' in file_path:
                if 'GT' not in file_path and 'copy' not in file_path:
                    train_imgs.append(file_path)
            if 'test' in file_path:
                if 'GT' in file_path:
                    test_masks.append(file_path)
                else:
                    test_imgs.append(file_path)

    train_imgs, val_imgs = train_test_split(np.array(train_imgs), test_size=0.20, random_state=69)
    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            ret.append(file.replace('.png', '_GT.png'))
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)


    if mode == 'semi-sup':
        return {
            'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
            'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]
        }
    else:
        return {
            'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]
        }


def get_neuseg(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    train_imgs = []
    test_imgs = []
    test_masks = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if 'train' in file_path:
                if 'images' in file_path:
                    train_imgs.append(file_path)
            if 'test' in file_path:
                if 'images' in file_path:
                    test_imgs.append(file_path)

    train_imgs, val_imgs = train_test_split(np.array(train_imgs), test_size=0.20, random_state=69)
    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            ret.append(file.replace('.jpg', '.png').replace('images', 'annotations'))
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)
    test_masks = get_mask(test_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)


    if mode == 'semi-sup':
        return {
            'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
            'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]
        }
    else:
        return {
            'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]
        }


def get_carpet(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    imgs = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if '.png' in file_path:
                imgs.append(file_path)


    train_imgs, test_imgs = train_test_split(np.array(imgs), test_size=0.20, random_state=69)

    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.25, random_state=69)

    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            ret.append(file.replace('.png', '.bmp'))
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)
    test_imgs = test_imgs.tolist()
    test_masks = get_mask(test_imgs)

    if mode == 'semi-sup':
        return {'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
                'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]}
    else:
        return {'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]}


def get_magnetic(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    imgs = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if '.jpg' in file_path and ('Blowhole' in file_path or 'Free' in file_path or 'Crack' in file_path or 'Fray' in file_path):
                imgs.append(file_path)


    train_imgs, test_imgs = train_test_split(np.array(imgs), test_size=0.20, random_state=69)

    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.25, random_state=69)

    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            ret.append(file.replace('.jpg', '.png'))
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)
    test_imgs = test_imgs.tolist()
    test_masks = get_mask(test_imgs)

    if mode == 'semi-sup':
        return {'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
                'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]}
    else:
        return {'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]}


def get_rsdd(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    imgs = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if 'Rail surface images' in file_path:
                imgs.append(file_path)


    train_imgs, test_imgs = train_test_split(np.array(imgs), test_size=0.20, random_state=69)

    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.25, random_state=69)

    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            ret.append(file.replace('Rail surface images', 'GroundTruth'))
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)
    test_imgs = test_imgs.tolist()
    test_masks = get_mask(test_imgs)

    if mode == 'semi-sup':
        return {'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
                'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]}
    else:
        return {'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]}


def get_severstal(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    imgs = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if 'train_images' in file_path and '.jpg' in file_path:
                imgs.append(file_path)


    train_imgs, test_imgs = train_test_split(np.array(imgs), test_size=0.20, random_state=69)

    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.25, random_state=69)

    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            ret.append(file.replace('train_images', 'train_masks').replace('jpg', 'png'))
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)
    test_imgs = test_imgs.tolist()
    test_masks = get_mask(test_imgs)

    if mode == 'semi-sup':
        return {'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
                'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]}
    else:
        return {'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]}


def get_cfd(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    imgs = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if 'Images' in file_path:
                imgs.append(file_path)


    train_imgs, test_imgs = train_test_split(np.array(imgs), test_size=0.20, random_state=69)

    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.25, random_state=69)

    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            f = file.replace('Images', 'Masks')
            f = f.replace('.jpg', '_label.PNG')
            ret.append(f)
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)
    test_imgs = test_imgs.tolist()
    test_masks = get_mask(test_imgs)

    if mode == 'semi-sup':
        return {'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
                'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]}
    else:
        return {'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]}


def get_cdd(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    imgs = []
    test_imgs, test_labels = [], []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if 'train_img' in file_path:
                imgs.append(file_path)
            if 'test_img' in file_path:
                test_imgs.append(file)
                f = file.replace('test_img', 'test_lab').replace('.jpg', '.png')
                test_labels.append(f)


    train_imgs, test_imgs = train_test_split(np.array(imgs), test_size=0.20, random_state=69)

    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.25, random_state=69)

    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            f = file.replace('train_img', 'train_lab')
            f = f.replace('.jpg', '.png')
            ret.append(f)
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)
    test_imgs = test_imgs.tolist()
    test_masks = get_mask(test_imgs)

    if mode == 'semi-sup':
        return {'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
                'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]}
    else:
        return {'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]}


def get_dagm(root_path, unlabeled_ratio=0.4, mode='semi-sup'):
    imgs = []
    test_imgs = []
    for root, dirs, files in os.walk(root_path):
        for file in files:

            file_path = os.path.join(root, file)
            if 'Train' in file_path and '.PNG' in file_path and 'Label' not in file_path:
                imgs.append(file_path)
            if 'Test' in file_path and '.PNG' in file_path and 'Label' not in file_path:
                test_imgs.append(file_path)


    train_imgs, val_imgs = train_test_split(np.array(imgs), test_size=0.2, random_state=69)

    labeled_imgs, unlabel_imgs = train_test_split(train_imgs, test_size=unlabeled_ratio, random_state=69)

    def get_mask(files):
        ret = []
        for file in files:
            file_name = file.split('\\')[-1]
            label_name = file_name.replace('.PNG', '_label.PNG')
            label_path = file.replace(file_name, f'Label\\{label_name}')
            if os.path.exists(label_path):
                ret.append(label_path)
            else:
                ret.append('0')
        return ret

    train_imgs = train_imgs.tolist()
    train_labels = get_mask(train_imgs)

    labeled_imgs = labeled_imgs.tolist()
    labeled_masks = get_mask(labeled_imgs)
    unlabel_imgs = unlabel_imgs.tolist()
    unlabel_masks = get_mask(unlabel_imgs)
    val_imgs = val_imgs.tolist()
    val_masks = get_mask(val_imgs)
    test_masks = get_mask(test_imgs)

    if mode == 'semi-sup':
        return {'imgs': [labeled_imgs, val_imgs, test_imgs, unlabel_imgs],
                'masks': [labeled_masks, val_masks, test_masks, unlabel_masks]}
    else:
        return {'imgs': [train_imgs, val_imgs, test_imgs], 'masks': [train_labels, val_masks, test_masks]}


if __name__ == '__main__':
    loaders = get_loaders('severstal', '', 4, 'total-sup')
    train_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']
    # train_loader.dataset.transform = A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_LINEAR)])
    # std, mean = train_loader.dataset.std, train_loader.dataset.mean
    device = torch.device('cpu')

    total_datas = []
    total_labels = []
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        # total_datas.append(img)
        # total_labels.append(label)
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        # total_datas.append(img)
        # total_labels.append(label)
    total_datas = torch.cat(total_datas, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    print(total_labels.max(), total_labels.min())
    for i in range(3):
        print(total_datas[:, i].mean(), total_datas[:, i].std())
