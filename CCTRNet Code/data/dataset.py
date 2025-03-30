import csv
import os
import re
import numpy as np
from torch.utils import data


class MyDataset(data.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, data_dir, csv_file_path):
        self.data_dir = data_dir
        # 数据集
        # os.listdir 返回一个包含目录中文件名的列表
        self.data_list = os.listdir(self.data_dir)
        self.csv_file_path = csv_file_path
        # 读取CSV文件到列表
        self.csv_data = []
        with open(self.csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            self.csv_data = list(reader)


    # 获取其中的每一条数据
    def __getitem__(self,item):
        file_path = self.data_dir + "/" + self.data_list[item]
        data = np.loadtxt(file_path, delimiter=',')
        # 使用正则表达式提取文件名中的序号
        match = re.match(r'(\d+)\.txt', self.data_list[item])
        if match:
            serial_number = int(match.group(1))
            # 查找 CSV 文件中对应的标签
            label = float(self.csv_data[serial_number][1])
            snr = float(self.csv_data[serial_number][2])
            sub_labels = float(self.csv_data[serial_number][3])
        
        return data,label,snr,sub_labels
    
    #返回数据集大小
    def __len__(self):
        return len(self.data_list)
    
