import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=4): #Change the number of classes-accordingly for each dataset
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(1, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)

            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def mDice(pred_mask, mask, smooth=1e-10, n_classes=4): #Change the number of classes occordingly for each dataset [4,7,6,2]
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        dice_per_class = []
        for clas in range(1, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                dice_per_class.append(np.nan)

            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                dice = 2*(intersect + smooth) / (union + intersect + smooth)
                dice_per_class.append(dice)
        return np.nanmean(dice_per_class)


class TotalDiceIou(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersect = np.zeros(num_classes)
        self.union = np.zeros(num_classes)
        self.correct_pixes = np.zeros(1)
        self.total_pixes = np.zeros(1)

    def reset(self):
        self.intersect = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.correct_pixes = np.zeros(1)
        self.total_pixes = np.zeros(1)

    def update(self, pred_mask, mask):
        pred_mask = pred_mask.flatten()
        mask = mask.flatten()

        for clas in range(1, self.num_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas
            # First calculate the intersection and concatenation of each batch and go through all the data in get_mIoU
            intersect = np.logical_and(true_class, true_label).sum()
            union = np.logical_or(true_class, true_label).sum()
            self.intersect[clas] += intersect
            self.union[clas] += union

        self.correct_pixes += np.sum(np.equal(mask, pred_mask))
        self.total_pixes += mask.size
        return

    def get_mIoU(self):
        smooth = 1e-10
        # Calculate the IoU for each class
        iou = (self.intersect[1:] + smooth) / (self.union[1:] + smooth)
        return iou.mean().item()

    def get_mdice(self):
        smooth = 1e-10
        dice = 2*(self.intersect[1:] + smooth) / (self.union[1:] + self.intersect[1:] + smooth)
        return dice.mean().item()

    def get_pixes_accuracy(self):
        acc = self.correct_pixes / self.total_pixes
        return acc.item()


class F1PR(object):
    def __init__(self):
        self.predicts = []
        self.targets = []
        self.results = {}

    def update(self, predict, target):
        with torch.no_grad():
            predict = torch.softmax(predict, dim=1)
            score = predict[:, 1]
            self.predicts.append(score)
            self.targets.append(target)
        return

    def reset(self):
        self.predicts = []
        self.targets = []
        self.results = {}

    def get_f1_pr(self):
        if isinstance(self.targets, list):
            self.targets = torch.cat(self.targets)
            self.predicts = torch.cat(self.predicts)
            self.targets = self.targets.cpu().numpy()
            self.predicts = self.predicts.cpu().numpy()

        # Calculate precision, recall and F1-score at all possible thresholds
        precisions, recalls, thresholds = precision_recall_curve(self.targets, self.predicts)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        auc_score = auc(recalls, precisions)
        best_f1_index = np.argmax(f1_scores)
        best_f1, best_r, best_p, thr = f1_scores[best_f1_index], recalls[best_f1_index], precisions[best_f1_index], thresholds[best_f1_index]
        self.results = {'precision': precisions, 'recall': recalls, 'f1': best_f1, 'thresholds': thresholds,
                        'auc': auc_score, 'best_r': best_r, 'best_p': best_p, 'threshold': thr}
        return self.results

    def draw(self, save_path):
        plt.clf()
        plt.figure(f"F1: {self.results['f1']}  thr: {self.results['threshold']}")
        plt.title(f"Recall: {self.results['best_r']} Precision: {self.results['best_p']}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(self.results['recall'], self.results['precision'])
        plt.savefig(save_path+'PR_Curve.png')
        np.save(save_path+'targets', self.targets)
        np.save(save_path+'predicts', self.predicts)
        return
