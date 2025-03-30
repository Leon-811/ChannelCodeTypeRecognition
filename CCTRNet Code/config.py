#coding:utf8
import warnings
class DefaultConfig(object):
    env = 'main' # visdom 环境
    model = 'ResNet34' # 使用的模型，名字必须与models/__init__.py中的名字一致
    train_data_root = './data/train/' # 训练集存放路径
    val_data_root = './data/val' # 测试集存放路径
    test_data_root = './data/test'
    data_csv = './data/data.csv'
    load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 512 # batch size
    device = "cup" # GPU id to use.
    mul_gpu = False

    num_workers = 4 # how many workers for loading data

    debug_file = './tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    max_epoch = 500
    lr = 0.1 # initial learning rate
    lr_decay = 0.5 # when val_loss increase, lr = lr*lr_decay
    lr_decay_patience = 5
    weight_decay = 0.1 
    earlystopping_patience = 25
    



    def parse(self,kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))


opt = DefaultConfig()
