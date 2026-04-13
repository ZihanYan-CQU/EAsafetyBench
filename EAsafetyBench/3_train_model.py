import os
import random
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method
from pathlib import Path
# 1. 获取脚本所在目录的绝对路径
root_dir = Path(__file__).resolve().parent
# 2. 切换工作目录
os.chdir(root_dir)

# 验证
print(f"当前工作目录已设置为: {os.getcwd()}")
# ==============================================================================
# ============================== 配置区域 (CONFIG) =============================
# ==============================================================================

# --- 1. 环境与设备 ---
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 指定使用的GPU
SEED = 403
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- 2. 路径与文件配置 ---
BASE_EXP_DIR = "./"
PROMPT_MODE = 'prompt_train'
FEATURE_BASE_DIR = os.path.join(BASE_EXP_DIR, f"train_feature/{PROMPT_MODE}")
MODEL_SAVE_BASE_DIR = os.path.join(BASE_EXP_DIR, "model")

# --- 3. 模型架构配置 (MLP) ---
MODEL_DIM = 2048       # 输入特征维度
HIDDEN_DIM_1 = 1024     # 第一层隐藏层
HIDDEN_DIM_2 = 512      # 第二层隐藏层
OUTPUT_DIM = 2           # 分类数
DROPOUT_RATE = 0.2       # Dropout 比例

# --- 4. 训练超参数 ---

#训练参数
batch_size = 16
lr = 0.001
weight_decay = 0.0002
epochs = 50

SCHEDULER_MILESTONES = [100, 100]
SCHEDULER_GAMMA = 0.1

# ==============================================================================
# ============================ 逻辑代码 (LOGIC) ===============================
# ==============================================================================

def is_directory_has_one_files(directory_path):
    """检查目录下文件数量是否符合预期"""
    if not os.path.exists(directory_path):
        return False
    return len(os.listdir(directory_path)) == 1

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ThreeLayerClassifier(nn.Module):  
    def __init__(self, dim, hidden1, hidden2, output_dim, dropout):
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x           
    
class TrainDataset(Dataset):
    def __init__(self, feature_path_list):
        self.feature = pd.concat(
            [torch.load(feature_path, weights_only=False) for feature_path in feature_path_list], 
            axis=0
        )

        # self.feature = self.feature.sample(n=20)
        # stratified_sample = self.feature.groupby('label').apply(lambda x: x.sample(n=100))
        # self.feature = stratified_sample.reset_index(drop=True)
        # self.features = self.features[self.features['label']!=4]
        
        #print(f"Dataset loaded, total samples: {len(self.feature)}")
        #print(self.feature.head())

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        data_col = self.feature.iloc[index, :]
        return data_col.tensor, torch.tensor(data_col.label, dtype=torch.long)

    def collate_fn(self, batch):
        batch_tensor = []
        batch_label = []
        for tensor, label in batch:
            batch_tensor.append(tensor.cuda())
            # 标签二值化处理
            batch_label.append(0 if label == 0 else 1)
            
        batch_tensor = torch.stack(batch_tensor)
        batch_label = torch.tensor(batch_label)
        return {'features': batch_tensor, 'labels': batch_label}

def get_loader(feature_path_list, batch_size, shuffle=False, num_workers=4):
    ds_df = TrainDataset(feature_path_list)
    dataloader_class = partial(DataLoader, pin_memory=False)
    loader = dataloader_class(
        ds_df, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=ds_df.collate_fn,
        num_workers=num_workers
    )
    return loader
    
def batch_train(model, dataloader, lr, weight_decay, epochs, model_save_path, device, layer_index):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 使用配置的调度器参数
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=SCHEDULER_MILESTONES, 
        gamma=SCHEDULER_GAMMA
    )
    
    train_losses = []
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        tloss = []
        bar = tqdm(dataloader)
        numbers = 0
        
        for index, batch in enumerate(bar):
            features = batch['features']
            labels = batch['labels']
            numbers += len(labels)
            
            optimizer.zero_grad()
            y_pred = torch.squeeze(model(features), 1)
            loss = criterion(y_pred.to(device), labels.to(device))
            
            loss.backward()
            tloss.append(loss.item())
            optimizer.step()
            
            bar.set_postfix(
                epoch=epoch, 
                learning_rate=optimizer.param_groups[0]['lr'], 
                loss=np.array(tloss).mean()
            )
        
        scheduler.step()
        print(f"Layer {layer_index} Epoch {epoch} processed samples: {numbers}")
        train_losses.append(np.array(tloss).mean())

    torch.save(model.state_dict(), model_save_path.format(epoch))

def train_for_layer_index(weight_decay, epochs, lr, batch_size, layer_indices):
    shuffle = True
    
    # 遍历当前进程负责的层
    for layer_index in layer_indices:
        # 构建路径 (使用 os.path.join)
        feature_path_list = [
            os.path.join(FEATURE_BASE_DIR, f'train_layer_{layer_index}.pt')
        ]
        
        # 模型保存路径模板
        model_save_dir = os.path.join(MODEL_SAVE_BASE_DIR, f"layer_{layer_index}")
        model_save_path = os.path.join(
            model_save_dir, 
            f"b{batch_size}_lr{'{:.0e}'.format(lr)}_e{{}}.pt"
        )

        # 目录检查与创建
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            print(f"--------------------------------------目录 '{model_save_dir}' 已创建。--------------------------------------")
        else:
            print(f"--------------------------------------目录 '{model_save_dir}' 已存在。--------------------------------------")

        # 检查是否已训练完成
        if is_directory_has_one_files(model_save_dir):
            print(f'layer {layer_index} 已完成训练，跳过...')
            continue

        # 初始化模型 (传入配置参数)
        model = ThreeLayerClassifier(
            dim=MODEL_DIM,
            hidden1=HIDDEN_DIM_1,
            hidden2=HIDDEN_DIM_2,
            output_dim=OUTPUT_DIM,
            dropout=DROPOUT_RATE
        )
        model.to(DEVICE)

        # 获取数据
        dataloader = get_loader(
            feature_path_list=feature_path_list,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

        # 开始训练
        #print(f"开始训练: Layer={layer_index}")
        batch_train(model, dataloader, lr, weight_decay, epochs, model_save_path, DEVICE,layer_index)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    
    # 设置随机种子
    set_seed(SEED)

    scan_folder = os.path.join(FEATURE_BASE_DIR)
    if os.path.exists(scan_folder):
        file_count = len([f for f in os.listdir(scan_folder) if os.path.isfile(os.path.join(scan_folder, f))])
        current_layer_indices = [i + 1 for i in range(file_count)]
        print(f"自动检测到 {file_count} 层待训练。")
    else:
        print(f"警告: 文件夹 {scan_folder} 不存在，无法自动获取层数。")

    # 拆分任务用于并行训练
    half = len(current_layer_indices) // 2
    layer_indices_part1 = current_layer_indices[:half]
    layer_indices_part2 = current_layer_indices[half:]

    # 多进程训练
    print(f"启动并行训练，进程数: 2")
    with Pool(processes=2) as pool:
        pool.starmap(
            train_for_layer_index, 
            [
                (weight_decay, epochs, lr, batch_size, layer_indices_part1), 
                (weight_decay, epochs, lr, batch_size, layer_indices_part2)
            ]
        )