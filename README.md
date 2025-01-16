# 表面缺陷分割通用仓库
这是一个适用于全监督分割任务的仓库。作者：郑宇博，邮箱1287293308@qq.com

## 1. 仓库结构
`main.py`: 执行主程序，所有的模型训练、测试均从该文件开始，内部包含了超参数设置，通过调用该程序并传参开始训练。

`data/dataloaders.py`: 数据流处理，支持多种数据集加载，也可以自定义设置。通过`benchmark`关键字索引数据集。

`model_trains/`: 包含多种模型的训练方式，主要是不同半监督学习模型之间训练方式有区别，因此需要单独设置，当然，某些全监督的模型也有训练方式的不同（比如损失函数优化）。
内部写了一个`basenetwork.py`的文件，该文件作为父类，定义了模型训练的初始化，验证、测试行为，故而当新的训练方式需要定义时，只需要继承它，重构训练行为即可。

`models/`: 模型结构的定义，目前以支持10+全监督分割模型，通过`net_factory`下的`get_model()`调用。

`utilities/`: 训练测试工具包，包括评估指标`metric.py`,梯度类激活图`gradcam.py`,记录训练过程`logger.py`,
多种损失函数`losses.py`,速度测试`fps.py`等。

## 2. 支持的数据集
目前已支持数据集有`NEU-SEG`，`KolektorSDD系列`等。

## 3. 支持的模型
### 3.1 全监督
`U-Net`, `Seg-Net`, `PGA-Net`, `DeepLabV3`, `BiseNet`, `EDR-Net`, `SegFormer`, `TopFormer`
### 3.2 半监督
`UAPS`, `Mean-Teacher`, `Consist`等


## 4. 环境需求
``Python >= 3.6 PyTorch >= 1.1.0 Albumentations tqdm tensorboardX cv2 numpy``
## 5. 运行
例如，我需要在NEU-SEG数据集上运行U-Net全监督模型，我需要执行语句(在终端Terminal中)输入：
`python main.py --model u_net --benchmark neuseg --base_lr 0.01 --epochs 100 --log_path logs --dataset_root_path C:/wrd/NEU_Seg --mode totat-sup --batch_size 4`

这段代码执行`main.py`程序，同时传入了许多参数，`--model`代表训练的模型，`--benchmark`代表数据集，`--dataset_root_path`代表的是下载数据集后存放的路径。

如果是在Pychram等IDE中执行，你可以手动修改`main.py`中的参数即可。你也可以使用`total_sup_batch.cmd`文件来实现运行一个程序后自动运行下一个。

程序运行逻辑：`main.py` --> `data/dataloader.py`数据集加载、实例化 --> `model_trains/`模型训练方式实例化 --> `models/`模型实例化 --> 
`model_trains/`完成训练验证测试过程 --> `utilities/`训练日志、指标评估、测试图片结果保存 --> `main.py`结束。
## 致谢

感谢所有支持本项目的朋友们

## Contact
wangyi@s.upc.edu.cn或532680475@qq.com
