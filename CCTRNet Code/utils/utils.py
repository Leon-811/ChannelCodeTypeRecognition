import os
import torch
import numpy as np
from torch import nn

class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""
    
    def __init__(self, patience=25, verbose=False, delta=0.001,env=None):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation accuracy improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        # self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0.0
        self.delta = delta
        self.env = env

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation accuracy increase.'''
        # path = os.path.join(self.save_path, 'best_network.pth')
        # torch.save(model.state_dict(), path)  # 保存当前最佳模型参数
        name = model.save(val_epoch_acc=val_acc,root=self.env)
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model:{name}')
        self.val_acc_max = val_acc




def try_gpu(i=0):  #@save
    """如果存在,则返回gpu(i),否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
    
# 计算 Kappa 系数
def calculate_kappa(cm):
    # 计算观察到的一致性 P_o
    total = np.sum(cm)
    P_o = np.trace(cm) / total

    # 计算预期的一致性 P_e
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    P_e = np.sum(row_sums * col_sums) / (total ** 2)

    # 计算 Kappa
    kappa = (P_o - P_e) / (1 - P_e)
    return kappa


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)