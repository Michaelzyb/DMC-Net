import argparse
import os
from datetime import datetime
import numpy as np
import torch
from data.dataloaders import get_loaders
from model_trains import *


def train_model(model_name):
    seed = 1337
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model_name, help='model_name',
                        choices=['dmcnet', 'u_net', 'bisenet', 'edrnet', 'deeplabv3', 'seg_net', 'seg_former', 'pga_net', 'bisenet', 'topformer'])
    parser.add_argument('--benchmark', type=str, default='neuseg', help='dataset',
                        choices=['KolektorSDD', 'KolektorSDD2', 'neuseg', 'carpet', 'hazelnut', 'MT', 'CrackForest', 'RSDD1',
                                 'Crack500', 'CDD', 'DAGM1', 'DAGM2', 'DAGM3', 'DAGM4', 'DAGM5', 'DAGM6', 'DAGM7', 'DAGM8', 'DAGM9', 'DAGM10'])
    parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_path', type=str, default='logs_test')
    parser.add_argument('--dataset_root_path', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--mode', type=str, default='total-sup', choices=['total-sup'])
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    logtime = datetime.now().__format__('_%m%d_%H%M%S')
    log_path = args.log_path+'/'+args.benchmark+'/'+args.model+logtime+'/'
    if args.mode == 'semi-sup':
        log_path = args.log_path + '/' + args.benchmark + '_' + str(args.unlabeled_ratio) + '/' + args.model + logtime + '/'

    # total-supervised model train
    if args.mode == 'total-sup':
        loaders = get_loaders(args.benchmark, args.dataset_root_path, args.batch_size, args.mode)
        NetWork = BaseLineTrain(args.epochs, args.benchmark, args.model, log_path, args.base_lr, args.checkpoint)
        NetWork.run(loaders['train'], unlabeled_loader=None, val_loader=loaders['val'], test_loader=loaders['test'], checkpoint=args.checkpoint)

if __name__ == '__main__':
    models_to_train = ['dmcnet']
    for model_name in models_to_train:
        print(model_name)
        train_model(model_name)
