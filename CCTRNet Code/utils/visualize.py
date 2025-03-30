#coding:utf8
import visdom
import time
import numpy as np

class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env,**kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        # self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    # def plot_many(self, d):
    #     '''
    #     一次plot多个
    #     @params d: dict (name,value) i.e. ('loss',0.11)
    #     '''
    #     for k, v in d.iteritems():
    #         self.plot(k, v)
    #
    # def img_many(self, d):
    #     for k, v in d.iteritems():
    #         self.img(k, v)

    def plot(self, win, title, epoch, train_value, val_value, **kwargs):
        '''
        self.plot('loss',1.00)
        win:窗口id
        name:线条名字
        '''
        # x = self.index.get(name, 0)
        self.vis.line(Y=np.array([train_value]), X=np.array([epoch]),
                      win=str(win),
                      opts=dict(title=title,xlabel="epoch",ylabel="Value"),
                      name="Train_value",
                      update='append',
                      **kwargs
                      )
        self.vis.line(Y=np.array([val_value]), X=np.array([epoch]),
                      win=str(win),
                      opts=dict(title=title,xlabel="epoch",ylabel="Value"),
                      name = "Val_value",
                      update = 'append',
                      **kwargs
                      )

    def heatmap(self,cm,snr):
        self.vis.heatmap(X=cm,  # 混淆矩阵
                        win=f'Confusion Matrix for SNR = {snr}', 
                        opts=dict(
                        title=f'Confusion Matrix for SNR = {snr}',
                        xlabel='Predicted Label',
                        ylabel='True Label',
                        colormap='Viridis'  # 可选的颜色映射
                        )
        )

    def plot_acc(self, win, acc,snr, **kwargs):
        '''
        self.plot('loss',1.00)
        win:窗口id
        name:线条名字
        '''
        # x = self.index.get(name, 0)
        self.vis.line(Y=np.array([acc]), X=np.array([snr]),
                      win=str(win),
                      opts=dict(title=win,xlabel="snr",ylabel="acc",showlegend=True),
                      name=str(win),
                      update='append',
                      **kwargs
                      )






    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(img_,
                        win=str(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''
        
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)
        
    def log_all(self, info, win='log_text'):
        
        log = ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(log, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
    
