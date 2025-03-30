#coding:utf8
import os
import torch as t
import time


class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None,val_epoch_acc=0,root=None,mul_gpu=False):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        checkpoint_root = f'checkpoints/{root}'  # 使用传入的 root 参数构建路径
        if checkpoint_root is not None:
            # 创建目录（如果不存在）
            os.makedirs(checkpoint_root, exist_ok=True)
            prefix = os.path.join(checkpoint_root, '')  # 将 checkpoint_root 添加到前缀中
        else:
            prefix = 'checkpoints/'  # 默认目录
            
        if name is None:
            # prefix = 'checkpoints/' + self.model_name + '_'
            formatted_val_acc = f"{val_epoch_acc:.4f}"  # 格式化为四位小数
            name = time.strftime(f"{prefix}{self.model_name}_val_acc_{formatted_val_acc}_%m%d_%H:%M:%S.pth")
            # name = time.strftime(prefix + "val_acc" + '_' + str(val_epoch_acc) +'_' +'%m%d_%H:%M:%S.pth')
        if mul_gpu:
            t.save(self.module.state_dict(), name)
        else:
            t.save(self.state_dict(), name)
        return name


class Flat(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)