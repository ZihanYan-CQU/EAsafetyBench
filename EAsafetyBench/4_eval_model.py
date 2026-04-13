import os
import time
import random
import glob
import yaml
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from functools import partial
from multiprocessing import Pool, set_start_method
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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
SEED = 403
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

# --- 2. 路径与文件配置 ---
BASE_EXP_DIR = "./"
TRAIN_FEATURE_DIR = os.path.join(BASE_EXP_DIR, "train_feature")
TEST_FEATURE_DIR = os.path.join(BASE_EXP_DIR, "test_feature")
MODEL_DIR = os.path.join(BASE_EXP_DIR, "model")
EVAL_DIR = os.path.join(BASE_EXP_DIR, "eval")
TEST_SAMPLE_CSV = os.path.join(BASE_EXP_DIR, "test_sample.csv")

EXPECTED_FILES_IN_EVAL_DIR = 6  # 检查是否完成评估的文件数

# --- 3. 模型架构配置 (MLP) ---
MODEL_DIM = 2048       # 输入特征维度 (ChatGLM通常是4096)
HIDDEN_DIM_1 = 1024     # 第一层隐藏层
HIDDEN_DIM_2 = 512      # 第二层隐藏层
OUTPUT_DIM = 2           # 分类数
DROPOUT_RATE = 0.7       # Dropout 比例 (注意：评估时Dropout会自动失效)

# --- 4. 评估超参数 ---
eval_batch_size = 128
train_batch_size = 16
lr = 0.001
weight_decay = 0.0002
epochs = [50]


# ==============================================================================
# ============================ 逻辑代码 (LOGIC) ===============================
# ==============================================================================

def process_mask(data_df):
    data_df['y_pred'] = data_df['predict'].apply(lambda x: 0 if x == 0 else 1)
    data_df['y_true'] = data_df['label_true'].apply(lambda x: 0 if x == 0 else 1)
    y_pred = list(data_df['y_pred'])
    y_true = list(data_df['y_true'])
    f1 = f1_score(y_true, y_pred)
    return f1

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def is_directory_has_six_files(directory_path):
    if not os.path.exists(directory_path):
        return False
    return len(os.listdir(directory_path)) == EXPECTED_FILES_IN_EVAL_DIR

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
    
class TestDataset(Dataset):
    def __init__(self, feature_path_list):
        self.feature = pd.concat(
            [torch.load(feature_path, weights_only=False) for feature_path in feature_path_list], 
            axis=0
        )

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        data_col = self.feature.iloc[index, :]
        return data_col.tensor, torch.tensor(data_col.label, dtype=torch.long)

    def collate_fn(self, batch):
        batch_tensor = []
        batch_label = [] # 修正了拼写 batct -> batch
        for tensor, label in batch:
            batch_tensor.append(tensor.cuda())
            # 标签二值化
            batch_label.append(0 if label == 0 else 1)
            
        batch_tensor = torch.stack(batch_tensor)
        batch_label = torch.tensor(batch_label)
        return {'features': batch_tensor, 'labels': batch_label}

def get_loader(feature_path_list, batch_size, shuffle=False, num_workers=4):
    ds_df = TestDataset(feature_path_list)
    dataloader_class = partial(DataLoader, pin_memory=False)
    loader = dataloader_class(
        ds_df, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=ds_df.collate_fn,
        num_workers=num_workers
    )
    return loader, ds_df.__len__()
    
def batch_eval(moderator, dataloader, eval_template, save_path):
    predict_list = []
    safe_prob_list = []
    unsafe_prob_list = []
    safe_num = 0
    unsafe_num = 0
    
    moderator.eval() # 确保是评估模式
    with torch.no_grad(): # 评估时关闭梯度计算，加速并节省显存
        for batch in tqdm(dataloader):
            # 增加 .to(DEVICE) 提高兼容性
            features = batch['features'].to(torch.float32).to(DEVICE)
            pred_scores = torch.squeeze(moderator(features), 1)
            
            for batch_index, pred_score in enumerate(pred_scores):
                pred = pred_score.argmax(dim=0)
                if pred == 1:
                    pred_label = 1
                    unsafe_num += 1
                else:
                    pred_label = 0
                    safe_num += 1
                
                pred_pro = F.softmax(pred_score, dim=0) 
                predict_list.append(pred_label)
                safe_prob_list.append(pred_pro[0].item())
                unsafe_prob_list.append(pred_pro[1].item())

    print("  safe: ", safe_num)
    print("unsafe: ", unsafe_num)
    
    eval_template['predict'] = predict_list
    eval_template['safe_prob'] = safe_prob_list
    eval_template['unsafe_prob'] = unsafe_prob_list
    
    f1 = process_mask(eval_template)
    print(f'---------------------------------------{save_path} F1-score:', f1)
    eval_template.to_csv(save_path, index=False)


def eval_for_layer_index(epochs, lr, train_batch_size, mode, layer_indices):
    time_list = []
    
    for layer_index in layer_indices:
        # 1. 构建评估结果保存路径
        eval_save_dir = os.path.join(EVAL_DIR, mode, f"layer_{layer_index}")
        
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
            print(f"目录 '{eval_save_dir}' 已创建。")
        else:
            print(f"目录 '{eval_save_dir}' 已存在。")

        # 检查是否已完成
        if is_directory_has_six_files(eval_save_dir):
            print(f'layer {layer_index} 已完成评估，跳过...')
            continue

        # 2. 遍历不同的 Epoch Checkpoint
        for epoch in epochs:
            moderator_name = f"b{train_batch_size}_lr{'{:.0e}'.format(lr)}_e{epoch}"

            # 构建各种路径
            eval_feature_path = os.path.join(TEST_FEATURE_DIR, mode, f"test_layer_{layer_index}.pt")
            save_csv_path = os.path.join(eval_save_dir, f"{moderator_name}.csv")
            model_load_path = os.path.join(MODEL_DIR, f"layer_{layer_index}", f"{moderator_name}.pt")

            # 3. 加载模型
            moderator = ThreeLayerClassifier(
                dim=MODEL_DIM,
                hidden1=HIDDEN_DIM_1,
                hidden2=HIDDEN_DIM_2,
                output_dim=OUTPUT_DIM,
                dropout=DROPOUT_RATE
            )
            
            # 加载权重
            state_dict = torch.load(model_load_path, map_location='cpu')
            moderator.load_state_dict(state_dict, assign=True)
            moderator.to(DEVICE)
            moderator.eval()

            # 4. 加载数据
            eval_template = pd.read_csv(TEST_SAMPLE_CSV)
            dataloader, test_len = get_loader(
                feature_path_list=[eval_feature_path],
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=0
            )

            # 5. 执行评估
            batch_eval(moderator, dataloader, eval_template, save_csv_path)
            
            # 清理显存
            del moderator
            torch.cuda.empty_cache()

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    seed_everything(SEED)

    # 主循环
    for mode in ['prompt_train','prompt_test']:
        print('='*80)
        print(f'------------------------------- Mode: {mode} -------------------------------')
                
        scan_folder = os.path.join(TRAIN_FEATURE_DIR,'prompt_train')
        if os.path.exists(scan_folder):
            file_count = len([f for f in os.listdir(scan_folder) if os.path.isfile(os.path.join(scan_folder, f))])
            current_layer_indices = [i + 1 for i in range(file_count)]
            print(f"自动检测到 {file_count} 层待评估。")
        else:
            print(f"警告: 文件夹 {scan_folder} 不存在，无法自动获取层数。")
            continue

        # 拆分任务
        half = len(current_layer_indices) // 2
        layer_indices_part1 = current_layer_indices[:half]
        layer_indices_part2 = current_layer_indices[half:]
        
        print(f"Parallel Part 1: {layer_indices_part1}")
        print(f"Parallel Part 2: {layer_indices_part2}")

        # 多进程评估
        with Pool(processes=2) as pool:
            pool.starmap(
                eval_for_layer_index, 
                [
                    (epochs, lr, train_batch_size, mode, layer_indices_part1), 
                    (epochs, lr, train_batch_size, mode, layer_indices_part2)
                ]
            )