#coding:utf8
import io
from config import opt
import os
import torch as t
import models
from data.dataset import MyDataset
from torch.utils.data import DataLoader
from utils.visualize import Visualizer
from utils.utils import calculate_kappa
from tqdm import tqdm
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from thop import profile
from ptflops import get_model_complexity_info
from collections import defaultdict

def test(**kwargs):
    opt.parse(kwargs)
    # 进入debug模式
    if os.path.exists(opt.debug_file):
        import ipdb;
        ipdb.set_trace()
    vis = Visualizer(opt.env)
    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path,)
    print("Use {} for testing".format(opt.device))
    model.to(t.device(opt.device))
    model.eval()  # 设置为评估模式
            
    macs, params = get_model_complexity_info(model, (2048,), as_strings=True,print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    macs_params_text = "<h2>Macs_Params</h2>"
    macs_params_text += "<p>"+'{:<30}  {:<8}'.format('Computational complexity: ', macs)+"<p>" 
    macs_params_text += "<p>"+'{:<30}  {:<8}'.format('Number of parameters: ', params)+"<p>"   
    vis.log_all(macs_params_text,'log_Macs_Params')

    # data
    test_data = MyDataset(opt.test_data_root,opt.data_csv)
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    # 假设 test_dataloader 是你的数据加载器
    all_predictions = {}  # 存储每个 SNR 的所有预测
    all_ground_truths = {}  # 存储每个 SNR 的所有真实标签

    for data, label, snr, _ in test_dataloader:
        # 假设 model 是你的模型，并且你已经定义了一个函数计算预测
        input = data.to(t.device(opt.device)).float()
        target = label.to(t.device(opt.device)).long()
        snr = snr.long()
        
        with t.no_grad():       
            outputs = model(input)  # 获取模型的输出
            predictions = outputs.argmax(dim=1)  # 假设是分类任务，获取预测标签

        # 批量处理每个样本的 SNR
        unique_snr = t.unique(snr)  # 获取当前 batch 中唯一的 SNR 值
        for current_snr in unique_snr:
            indices = (snr == current_snr).nonzero(as_tuple=True)[0]  # 获取当前 SNR 对应的样本索引

            # 收集当前 SNR 的预测和真实标签
            if current_snr.numpy() not in np.unique(np.array(list(all_predictions.keys()))):
                all_predictions[int(current_snr.numpy())] = []
                all_ground_truths[int(current_snr.numpy())] = []

            all_predictions[int(current_snr.numpy())].extend(predictions[indices].cpu().numpy())  # 将预测移回 CPU 并存储
            all_ground_truths[int(current_snr.numpy())].extend(target[indices].cpu().numpy())  # 将真实标签移回 CPU 并存储


    # 计算每个 SNR 的混淆矩阵和准确率
    snr_accuracy = {}
    confusion_matrices = {}
    for snr in all_predictions.keys():
        cm = confusion_matrix(all_ground_truths[snr], all_predictions[snr])  # 计算混淆矩阵
        confusion_matrices[snr] = cm
        
        # 在 Visdom 中绘制混淆矩阵
        # vis.heatmap(cm,snr)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
        disp.plot()
        plt.title(f'Confusion Matrix for SNR {snr}')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')  # 保存为图像文件
        buf.seek(0)
        # 将图像读取为 NumPy 数组并转换
        image = plt.imread(buf)
        image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W)
        vis.img(f"confusion_matrix@{snr}",image.astype(np.float32))
        
        # 计算准确率
        correct_predictions = np.trace(cm)  # 混淆矩阵对角线上的元素是正确预测数量
        total_predictions = np.sum(cm)  # 混淆矩阵中所有元素的总和是总预测数量
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        snr_accuracy[snr] = accuracy
        
    accuracy_text = "<h2>SNR Accuracy</h2>"
    total_acc = 0
    for snr, accuracy in snr_accuracy.items():
        accuracy_text += f"<p>SNR: {snr}, Accuracy: {accuracy:.4f}</p>"
        total_acc += accuracy
        # print((snr,accuracy))
        vis.plot_acc(win="Acc_SNR Curve", acc=accuracy,snr=snr)
    accuracy_text += f"<p>Average Accuracy: {total_acc/16:.4f}</p>"    
            
    # 初始化整体混淆矩阵
    overall_confusion_matrix = np.zeros_like(next(iter(confusion_matrices.values())))

    
    
    # 合并混淆矩阵
    for cm in confusion_matrices.values():
        overall_confusion_matrix += cm
        
        
    kappa_text = "<h2>Kappa Coefficient</h2>"
    kappa_value = calculate_kappa(overall_confusion_matrix)
    kappa_text += f"<p>Kappa Coefficient: {kappa_value}</p>"    
    vis.log_all(kappa_text,'log_Kappa Coefficient')
    
    # vis.heatmap(overall_confusion_matrix,snr="all")
    overall_disp = ConfusionMatrixDisplay(confusion_matrix=overall_confusion_matrix, display_labels=[0, 1, 2, 3, 4])
    overall_disp.plot()
    plt.title('Overall Confusion Matrix')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')  # 保存为图像文件
    buf.seek(0)
    # 将图像读取为 NumPy 数组并转换
    image = plt.imread(buf)
    image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W)
    vis.img(f"confusion_matrix @ all ",image.astype(np.float32))
    
    cm_text = "<h2>Confusion Matrix</h2>"
    for snr, cm in confusion_matrices.items():
            cm_text += f"<p>SNR: {snr}, Confusion Matrix: {cm}</p>"
    cm_text += f"<p>SNR: all, Confusion Matrix: {overall_confusion_matrix}</p>"
    
    # 现在 snr_accuracy 字典中存储了每个 SNR 的准确率
    # write_csv(snr_accuracy,opt.result_file)
    vis.log_all(accuracy_text,'log_snr_accuracy')
    vis.log_all(cm_text,'log_snr_confusion_matrix')

    # 假设你的 dataloader 返回：
    # inputs: 测试数据
    # labels: 大类类别编号（0-4）
    # sub_labels: 该大类下的小类类别编号

    # 初始化存储正确预测数和总数的字典
    num_classes = {
        0: 4,  # BCH
        1: 4,  # Conv
        2: 12, # Polar
        3: 12, # LDPC
        4: 1   # Turbo
    }

    # 记录每个小类的正确数和总数
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    # 运行测试
    with t.no_grad():
        for inputs, labels, _, sub_labels in test_dataloader:
            inputs = inputs.to(t.device(opt.device)).float()
            labels = labels.to(t.device(opt.device)).long()
            sub_labels = sub_labels.to(t.device(opt.device)).long()

            # 获取模型预测
            outputs = model(inputs)  # 假设模型输出 shape: [batch_size, 总类别数]
            _, predicted = t.max(outputs, 1)  # 获取预测的大类类别编号

            # 统计正确预测数
            for i in range(len(labels)):
                true_class = labels[i].item()
                true_subclass = sub_labels[i].item()-1
                pred_class = predicted[i].item()

                # 只有当大类预测正确时，才统计小类的正确率
                if pred_class == true_class:
                    correct_counts[(true_class, true_subclass)] += 1
                total_counts[(true_class, true_subclass)] += 1

    # 计算并格式化 HTML 输出
    accuracy_text = "<h2>Test Accuracy by Subclass</h2>"

    for major_class in num_classes.keys():
        accuracy_text += f"<h3>Class {major_class}</h3><ul>"
        for sub_class in range(num_classes[major_class]):
            total = total_counts[(major_class, sub_class)]
            correct = correct_counts[(major_class, sub_class)]
            acc = correct / total if total > 0 else 0.0
            accuracy_text += f"<li>Subclass {sub_class}: Accuracy = {acc:.4f}</li>"
        accuracy_text += "</ul>"

    # 记录日志
    vis.log_all(accuracy_text, "log_Test Accuracy")
