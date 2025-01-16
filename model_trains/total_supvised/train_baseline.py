from datetime import datetime
from model_trains.basenetwork import BaseNetWork
import torch


class BaseLineTrain(BaseNetWork):
    def __init__(self, epochs, benchmark, method, log_path, lr, checkpoint):
        super().__init__(epochs, benchmark, method, log_path, lr, checkpoint)

    def epoch_train(self):
        running_ce_loss = 0.0
        running_dice_loss = 0.0
        running_train_loss = 0.0

        self.optimizer.zero_grad()
        self.model.train()
        weight = [1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        for i, (inputs_S1, labels_S1) in enumerate(self.train_loader):
            inputs_S1, labels_S1 = inputs_S1.to(self.device), labels_S1.to(self.device)
            # Labeled samples output
            outputs = self.model(inputs_S1)
            loss_ce, loss_dice = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)
            if isinstance(outputs, (tuple, list)):
                for index in range(len(outputs)):
                    loss_ce += self.ce_loss(outputs[index], labels_S1.long()) * weight[index]
                    if self.num_classes == 2:
                        loss_dice += self.dice_loss(labels_S1.unsqueeze(1), outputs[index]) * weight[index]
                outputs = outputs[0]
            else:
                # CE_loss
                loss_ce = self.ce_loss(outputs, labels_S1.long())
                if self.num_classes == 2:
                    # Dice_loss
                    loss_dice = self.dice_loss(labels_S1.unsqueeze(1), outputs)
            # total supervised loss
            total_loss = 0.5 * (loss_ce + loss_dice)
            total_loss.backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_train_loss += total_loss.item()
            running_ce_loss += loss_ce.item()
            running_dice_loss += loss_dice.item()
            self.metric.update(torch.argmax(outputs, dim=1).cpu().numpy(), labels_S1.cpu().numpy())

        for param_group in self.optimizer.param_groups:
            lr_ = param_group['lr']
            break
        self.scheduler.step()

        epoch_loss = running_train_loss / len(self.train_loader)
        epoch_ce_loss = running_ce_loss / len(self.train_loader)
        epoch_dice_loss = running_dice_loss / len(self.train_loader)
        epoch_iou = self.metric.get_mIoU()
        epoch_dice = self.metric.get_mdice()
        epoch_pixes_acc = self.metric.get_pixes_accuracy()
        self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                         format(datetime.now(), self.epoch, self.total_epochs, epoch_loss))

        self.logger.info('Train loss: {}'.format(epoch_loss))
        self.writer.add_scalar('Train/Loss', epoch_loss, self.epoch)

        self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
        self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, self.epoch)

        self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
        self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, self.epoch)
        self.logger.info('Train IoU: {}'.format(epoch_iou))
        self.writer.add_scalar('Train/IoU', epoch_iou, self.epoch)
        self.logger.info('Train Dice: {}'.format(epoch_dice))
        self.writer.add_scalar('Train/Dice', epoch_dice, self.epoch)
        self.logger.info('Train Pixes Accuracy: {}'.format(epoch_pixes_acc))
        self.writer.add_scalar('Train/Pixes_acc', epoch_pixes_acc, self.epoch)
        self.logger.info('Lr: {}'.format(lr_))
        self.writer.add_scalar('info/lr', lr_, self.epoch)
        return
