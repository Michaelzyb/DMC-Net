# DMC-Net
Author: Yubo Zheng, email:1287293308@qq.com

## 1. Warehouse structure
`main.py`: Execute the main program, all model training and testing starts from this file, which contains the hyperparameter settings internally, 
           and training is started by calling this program and passing the parameters.

`data/dataloaders.py`: Data stream processing with support for multiple dataset loading and also customizable settings. Indexing of datasets by `benchmark` keyword.
`model_trains/`: Includes training methods for multiple models, mainly because there are differences in training methods between different semi-supervised learning models, so they need to be set up separately, and of course, some fully supervised models also have differences in training methods (e.g. loss function optimization).
Internally, a `basenetwork.py` file is written, which serves as a parent class that defines the initialization, validation, and testing behaviors of the model training, so when a new training method needs to be defined, you just need to inherit it and refactor the training behaviors.

`models/`: Definition of model structure, currently supporting 10+ fully supervised segmentation models, called via `get_model()` under `net_factory`.

`utilities/`: Training test toolkit, including evaluation metrics `metric.py`, gradient class activation graphs `gradcam.py`, logging training process `logger.py`,
Multiple loss functions `losses.py`, speed tests `fps.py`, etc.

## 2. Supported datasets
Currently supported datasets are `NEU-SEG`, `KolektorSDD series` and others.

## 3. Supported Models
`U-Net`, `Seg-Net`, `PGA-Net`, `DeepLabV3`, `BiseNet`, `EDR-Net`, `SegFormer`, `TopFormer`,`DMC-Net`.



## 4. Environmental needs
``Python >= 3.6 PyTorch >= 1.1.0 Albumentations tqdm tensorboardX cv2 numpy``
